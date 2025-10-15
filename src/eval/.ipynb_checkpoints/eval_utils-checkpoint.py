import pandas as pd
import pickle as pkl
import json
import random
import re
from tqdm import tqdm
from rapidfuzz import process
from data import coca_utils, presidential_utils, congress_utils
from data.fsm_verses import mask_verse_quotes_in_sermon
from data.data_utils import get_verses_from_sermon, get_verses_from_sermon_without_context, load_spacy_sentencizer, load_sermons
from nltk.tokenize import sent_tokenize


VERSE_REGEX = r"\b(?:[1-3]?\s?[A-Za-z]+(?:\s?[1-3]?[0-9]))(?::\s?[1-9][0-9]*(?:-[1-9][0-9]*)?)?\b"


def load_paired_data(cv_path, split, cw, verse_text_dict, verse_to_idx, num_examples, easy=False, pop_verses=False, input_is_text_not_cite=False):
    commentary_verse_df = pd.read_csv(cv_path + split +
                                      '/all_' + str(cw[0]) + 'before' + str(cw[1]) + 'after.csv')
    print("Removing instances where verse is unknown...")
    commentary_verse_df = commentary_verse_df[commentary_verse_df['verse'] != "UNKNOWN"]

    # filtering CV df to only include verses where the verse is in the verse-text dictionary
    commentary_verse_df = commentary_verse_df[commentary_verse_df['verse'].apply(lambda x: x in verse_text_dict.keys())]
    commentary_verse_df = commentary_verse_df.sample(num_examples)

    if easy:
        commentary_verse_df = commentary_verse_df[commentary_verse_df['verse'].apply(lambda x: x in pop_verses)]

    all_masked_quotes = zip(commentary_verse_df['verse'].apply(lambda x: verse_to_idx[verse_text_dict[x]]),
                                commentary_verse_df['verse'])
    all_contexts = commentary_verse_df['context'].apply(lambda x: x.split('[VERSE]'))  # outputs list of list of pairs
    return commentary_verse_df, all_masked_quotes, all_contexts


def load_whole_sermon_data(path, split, verse_text_dict, verse_to_idx, num_examples, fuzzy_citation_dict, input_is_text_not_cite=False, easy=False, pop_verses=False):
    if easy:
        verse_text_dict = {k: v for k,v in verse_text_dict.items() if k in pop_verses}
    
    # i.e., remove verses where...
    # the citation is not in the set of canonical verse citations
    # the verse has been previously labeled as unknown (likely because it wasn't a verse after all)
    def remove_unknown_verses(v):
        return [e for e in v if (e in verse_text_dict.keys()) and (e != "UNKNOWN") and ((not easy) or e in pop_verses)]
    
    import spacy
    from spacy.pipeline import Sentencizer
    nlp = spacy.load("en_core_web_sm")
    config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
    nlp.add_pipe("sentencizer", config=config)
    
    def sent_tok(sermon_text):
        assert (nlp is not None)
        docs = nlp(str(sermon_text), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        return [t.text for t in docs.sents]

    sermon_df = pd.read_csv(path + split + '.csv')
    sermon_df['verses'] = sermon_df['text'].apply(sent_tok).apply(lambda x: get_verses_from_sermon_without_context(x, fuzzy_citation_dict))
    # returns a series of dfs for each sermon: 'verse' column and 'sent_idx' column
    sermon_df['verses'] = sermon_df['verses'].apply(remove_unknown_verses)

    if num_examples != -1:
        print("Sampling because num_examples is not -1...")
        sermon_df = sermon_df.sample(num_examples)

    print("Now creating sermon verse lists...")
    print(sermon_df.head())
    print(sermon_df['verses'].sample(4))
    sermon_verse_lists = zip(sermon_df['verses'].apply(lambda x: [verse_to_idx[verse_text_dict[e]] for e in x]), sermon_df['verses'])
    sermon_texts = sermon_df['text']
    return sermon_df, sermon_verse_lists, sermon_texts


def load_ref_data(path, sermon_df, verse_text_dict, verse_to_idx, num_examples, fuzzy_citation_dict, input_is_text_not_cite=False, easy=False, pop_verses=False, load_nulls=False):
    if easy:
        verse_text_dict = {k: v for k,v in verse_text_dict.items() if k in pop_verses}
        text_verse_dict = {v: k for k,v in verse_text_dict.items()}
    
    def remove_unknown_verses(v):
        return [e for e in v if (e in verse_text_dict.keys()) and (e != "UNKNOWN") and ((not easy) or e in pop_verses)]
    
    import spacy
    from spacy.pipeline import Sentencizer
    nlp = spacy.load("en_core_web_sm")
    config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
    nlp.add_pipe("sentencizer", config=config)
    
    def sent_tok(sermon_text):
        assert (nlp is not None)
        docs = nlp(str(sermon_text), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        return [t.text for t in docs.sents]

    sermon_data = pd.read_csv(path)

    verse_list = list(sermon_data['verse'])
    texts = list(sermon_data['text'])

    if num_examples != -1:
        print("Sampling because num_examples is not -1...")
        idx = random.sample(range(0, len(verse_list)))
        verse_list = verse_list[idx]
        texts = texts[idx]

    sermon_verse_lists = []
    sermon_text_final = []

    # later
    print("* DEBUG *")
    print("verse to idx: ")
    print(verse_to_idx.keys())
    
    for i,t in zip(verse_list, texts):
        if verse_text_dict[i] in verse_to_idx.keys():
            sermon_verse_lists.append((verse_to_idx[verse_text_dict[i]], i))
            sermon_text_final.append(t)
        #else:
            #print(verse_text_dict[i])
    return sermon_verse_lists, sermon_text_final


def create_negative_examples_coca(coca_root_dir='/data/laviniad/COCA/', output_path_prefix='/data/laviniad/sermons-ir/negative_reference_examples/coca', num_examples=10000, context_length=100):
    keywords = ["God", "Jesus", "Christ"]
    excluded_keywords = ["Bible", "Scripture"]

    year_genre_dict = coca_utils.load_COCA_from_raw(coca_root_dir, join_text=False) # is year --> genre --> text list
    examples = []
    orig_pattern = "|".join(map(re.escape, keywords))
    excluded_pattern = "|".join(map(re.escape, excluded_keywords))
    pattern = f"(?i)({orig_pattern})"
    
    print("Finding matches")
    for m,genre_dict in year_genre_dict.items():
        for genre in genre_dict.keys():
            if genre == 'newspaper_lsp' or genre == 'magazine_qch': # neutral enough
                for f in tqdm(genre_dict[genre]):
                    match = re.search(pattern, f)
                    temp = [] # i think it's more efficient to mutate the smaller list (don't exactly remember how python does this, though)
                    if match:
                        start_index = max(0, match.start() - context_length)
                        end_index = min(len(f), match.end() + context_length)

                        context = f[start_index:end_index]
                        split_context = context.split('. ')
                        for idx,sent in enumerate(split_context):
                            if idx > 0 and idx < len(split_context) - 1: # not end or beginning -- so likely full sentence
                                if "Bible" not in sent and "Scripture" not in sent:
                                    temp.append(sent + '.')
                    
                    examples += temp
    
    # sampling
    print(f"Sampling and dumping after finding {len(examples)} examples")
    examples = random.sample(examples, min(num_examples, len(examples)))
    with open(output_path_prefix + '_' + str(len(examples)) + '.json', 'w') as f:
        json.dump(examples, f)
        print(f"Dumped to {f.name}")
        
        
def contains_citation(text_string):
    matches = re.findall(VERSE_REGEX, text_string)
    return ((matches == []) or (matches == None))
    

def create_negative_examples_sermons(output_path_prefix='/data/laviniad/sermons-ir/negative_reference_examples/sermons', num_examples=10000):
    sermons = load_sermons() # is from with_columns.csv -- i.e., unmodified
    examples = []
    
    for idx,row in sermons.iterrows():
        text = row['text']
        sents = sent_tokenize(text)
        sents = [i for i in sents[:max(len(sents),20)] if not contains_citation(i)]
        examples += sents[:min(len(sents), 10)]
    
    examples = random.sample(examples, min(len(examples), num_examples))
    with open(output_path_prefix + '_' + str(len(examples)) + '.json', 'w') as f:
        json.dump(examples, f)
        print(f"Dumped to {f.name}")
        

def create_negative_examples_congress(output_path_prefix='/data/laviniad/sermons-ir/negative_reference_examples/congress', nonprocedural_indices_path="/data/laviniad/congress_errata/nonprocedural_indices.json", 
                                      congress_path='/data/corpora/congressional-record/', num_examples=10000):
    with open(nonprocedural_indices_path) as f:
        nonprocedural_indices = json.load(f)
    
    weights = congress_utils.load_procedural_weights()
    congress_df_full = congress_utils.load_full_df_from_raw(congress_path)
    examples = []
    
    for idx, row in congress_df_full.iterrows():
        text = row['text']
        if not congress_utils.filter_congress_with_procedural_classifier(text, weights): # looking for procedural text
            sents = sent_tokenize(text)
            examples += sents[:min(len(sents), 10)]
        
    examples = random.sample(examples, min(len(examples), num_examples))
    with open(output_path_prefix + '_' + str(len(examples)) + 'congress.json', 'w') as f:
        json.dump(examples, f)
        print(f"Dumped to {f.name}")
        

def create_negative_examples():
    #create_negative_examples_coca()
    #create_negative_examples_sermons()
    create_negative_examples_congress()
    print("Created the negative example files")


def load_negative_examples(negative_examples_path='/data/laviniad/sermons-ir/negative_reference_examples/', num_examples=10000):
    examples = []
    try:
        with open(negative_examples_path + 'coca_' + str(num_examples) + '.json') as f:
            coca_list = json.load(f)
            examples += coca_list
        
        with open(negative_examples_path + 'sermons_' + str(num_examples) + '.json') as f:
            sermon_list = json.load(f)
            examples += sermon_list
        
        with open(negative_examples_path + 'congress_' + str(num_examples) + '.json' + str(num_examples) + '.json') as f:
            sermon_list = json.load(f)
            examples += sermon_list
        
    except FileNotFoundError:
        print("Could not find negative example files, returning empty example list")
        return []
    
    return examples
           

def load_json_sermon_references(path, verse_text_dict, verse_to_idx, fuzzy_citation_dict, input_is_text_not_cite=False, easy=False, pop_verses=False, load_nulls=False, load_external_negatives=False, mask_verse_text=True, coca_path='/data/laviniad/COCA/', sermon_path='/data/laviniad/sermons-ir/sermons_clean_text.csv'):
    if easy:
        verse_text_dict = {k: v for k,v in verse_text_dict.items() if k in pop_verses}
        text_verse_dict = {v: k for k,v in verse_text_dict.items()}
        
    import spacy
    from spacy.pipeline import Sentencizer
    nlp = spacy.load("en_core_web_sm")
    config = {"punct_chars": ['!', '.', '?', '...']}
    nlp.add_pipe("sentencizer", config=config)
    
    def sentencize(sermon_text):
        doc = nlp(str(sermon_text), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        return [t.text for t in doc.sents]
    
    print('the path is {}'.format(path))
    with open(path, 'r') as f:
        data = json.load(f)
        
    print('now making reflist')
    ref_list = []
    for sermon_ref_obj in data:
        for o in sermon_ref_obj[1]: 
            ref_list.append(o)

    verse_list = []
    text_list = []
    label_list = []
    for ref in tqdm(ref_list):
        assert(isinstance(ref['verse'], str))
        v = ref['verse']
            
        if v in verse_text_dict.keys() and v in verse_to_idx.keys():
            orig = mask_verse_quotes_in_sermon(ref['original_text'], v, sentence_tokenizer=sentencize)
            
            verse_list.append(v)
            text_list.append(orig)
            label_list.append(verse_to_idx[v])
        elif load_nulls:
            orig = mask_verse_quotes_in_sermon(ref['original_text'], v, sentence_tokenizer=sentencize)
            
            verse_list.append('NULL')
            text_list.append(orig)
            label_list.append(-1)
            
    if load_external_negatives:
        neg_verse_list, neg_text_list, neg_label_list = load_negative_examples(label_schema)
        verse_list += neg_verse_list
        text_list += neg_text_list
        label_list += neg_label_list

    return verse_list, text_list, label_list
