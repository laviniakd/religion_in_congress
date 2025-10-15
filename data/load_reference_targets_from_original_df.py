import pandas as pd
import json
import pandas as pd
import re
from rapidfuzz import fuzz
from ast import literal_eval
import argparse
from rapidfuzz.process import extractOne
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle as pkl

from nltk.tokenize import sent_tokenize

from bs4 import BeautifulSoup
import re

from data.bible_utils import bible_helper
from data.data_utils import verse_range_to_individual_list, load_spacy_sentencizer, get_verse_dicts

VERSE_REGEX = r'\b(?:\d\s\w+\s\d+:\d+-\d+|\d\s\w+\s\d+:\d+|\w+\s\d+:\d+-\d+|\w+\s\d+:\d+)\b'

QUOTE_PATTERN = re.compile(r'\"[^\"]*\"')
LINK_PATTERN = r'((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)'
HTML_PATTERN = r"<([a-z]+)(?![^>]*\/>)[^>]*>"


def return_cleaning_string_patterns():
    return VERSE_REGEX, QUOTE_PATTERN, LINK_PATTERN, HTML_PATTERN


BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/bibles/AmericanKJVBible.txt'
FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'


# remove links and tags + normalize whitespace
# should also send unicode characters to normal versions?
def clean(t):
    no_link = re.sub(LINK_PATTERN, '', t)
    no_link_or_html = re.sub(HTML_PATTERN, '', no_link)
    normalized = re.sub(" {2,}", " ", no_link_or_html.strip())
    normalized = re.sub("\n{2,}", "\n", normalized)
    
    return normalized


def get_matched_citations(sermon_idx, sent_toked_sermon, verse_to_text, fuzzy_citation_dict):
    sermon_quals = []
    
    for s in sent_toked_sermon:
        match = re.match(VERSE_REGEX, s[1])
        # need to know what match returns...
    
        if match:
            verse = match.group(0)
            if verse in fuzzy_citation_dict.keys():
                verse = fuzzy_citation_dict[verse]

                if verse in verse_to_text.keys():
                    sermon_quals.append({'original_text': s[1], 
                                         'sermon_idx': sermon_idx, 
                                         'verse': verse,
                                         'verse_text': verse_to_text[verse],
                                         'type': 'citation',
                                         'sentence_number_in_doc': s[0]
                                        })
                
    return sermon_quals


def get_matched_verses(sermon_idx, sent_toked_sermon, text_to_verse, verses, length_minimum=5, threshold=80):
    sermon_quals = []
    
    for s in sent_toked_sermon:
        choice_v = None
        if len(s[1].split()) >= length_minimum:
            choice_v = extractOne(s[1], verses)

        if choice_v is not None and choice_v[1] > threshold:
            sermon_quals.append({'original_text': s[1], 
                                 'sermon_idx': sermon_idx, 
                                 'verse': text_to_verse[choice_v[0]],
                                 'verse_text': choice_v[0],
                                 'type': 'verse_text',
                                 'sentence_number_in_doc': s[0]
                                })

    return sermon_quals


# TODO: implement more principled way of recognizing sermon introductions
def recognize_intro(sent_toked_sermon):
    pass


def retrieve_whole_sermon_verse_labels(enum_sermon_chunk, dumb_intro=True):
    nlp = load_spacy_sentencizer(punct_chars=['!', '.', '?', '...'])
    idxs = [i[0] for i in enum_sermon_chunk]
    texts = [i[1] for i in enum_sermon_chunk]
    verses = [literal_eval(i[2]) for i in enum_sermon_chunk]
    verses = [[s.lower() for s in v] for v in verses]
    #print(f"verses: {verses}")
    
    count = 0
    sentence_tokenized_sermons = []
    
    for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
        sentence_tokenized_sermons.append([s.text.strip() for s in doc.sents if s.text.strip()])
        count += 1
    
    results = []
    for i, sent_toked_sermon, verse_list in tqdm(zip(idxs, sentence_tokenized_sermons, verses), total=len(sentence_tokenized_sermons)):
        if dumb_intro:
            intro = ' '.join(sent_toked_sermon[:DUMB_INTRO_LENGTH])
        else:
            intro = recognize_intro(sent_toked_sermon)
        
        sermon_quals = []
        for f in verse_list:
            if f in FUZZY_CITATION_DICT.keys():
                verse_canonical = FUZZY_CITATION_DICT[f]
            else:
                verse_canonical = f
               
            if verse_canonical in VERSE_TO_TEXT.keys():
                text_of_verse = VERSE_TO_TEXT[verse_canonical]

                sermon_quals.append({'original_text': intro, 
                                         'sermon_idx': i, 
                                         'verse': verse_canonical,
                                         'verse_text': text_of_verse,
                                         'type': 'whole_sermon',
                                         'sentence_number_in_doc': 'multi_sentence'
                                        })
        results.append((i, sermon_quals))
    
    return results


# retrieve citations for chunk of sermons
def retrieve_references(enum_sermon_chunk):
    nlp = load_spacy_sentencizer()
    idxs = [i[0] for i in enum_sermon_chunk]
    texts = [i[1] for i in enum_sermon_chunk]

    sentence_tokenized_sermons = []
    count = 0
    for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
        sentence_tokenized_sermons.append([(count, s.text.strip()) for s in doc.sents if s.text.strip()])
        count += 1
        #if count % 50 == 0:
            #print(f'{count} instances processed')

    reference_list = []

    #print("Debugging")
    #print(f"Sentence-tokenized sermons: {len(sentence_tokenized_sermons)}")
    
    for i, sent_toked_sermon in tqdm(zip(idxs, sentence_tokenized_sermons), total=len(sentence_tokenized_sermons)):
        citation_refs = get_matched_citations(i, sent_toked_sermon, VERSE_TO_TEXT, FUZZY_CITATION_DICT)
        cited_verse_texts = set([c['verse_text'] for c in citation_refs])
        text_refs = get_matched_verses(i, sent_toked_sermon, TEXT_TO_VERSE, set(verse_set) | set(cited_verse_texts), threshold=args.fsm_threshold)
        ref_list = citation_refs + text_refs
        reference_list.append((i, ref_list))
    
    return reference_list

if __name__ == '__main__':
    bible_df = bible_helper(BIBLE_PATH)
    VERSE_TO_TEXT, FUZZY_CITATION_DICT = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
    FUZZY_CITATION_DICT = {k.lower(): v.lower() for k,v in FUZZY_CITATION_DICT.items()}
    VERSE_TO_TEXT = {k.lower(): v.lower() for k,v in VERSE_TO_TEXT.items()}
    TEXT_TO_VERSE = {v: k for k,v in VERSE_TO_TEXT.items()}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--sermon_path', default='/data/laviniad/sermons-ir/sermoncentral/sermons_clean_text.csv', type=str)
    parser.add_argument('--output_dir', default='/data/laviniad/sermons-ir/sermoncentral/references/', type=str)
    parser.add_argument('--only_gold_verses', action='store_true')
    parser.add_argument('--fsm_threshold', default=90, type=int)
    parser.add_argument('--num_verses', default=-1, type=int)
    parser.add_argument('--dumb_intro_length', default=5, type=int)
    parser.add_argument('--chunk_size', default=1000, type=int)
    parser.add_argument('--clean_text', action='store_true')
    #parser.add_argument('--popular_verses', default=-1, type=int)
    # ^ implement above -- if -1 is just all verses
    args = parser.parse_args()
    
    DUMB_INTRO_LENGTH = args.dumb_intro_length
    POPULAR_VERSES = '/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv'
    popular_verses = pd.read_csv(POPULAR_VERSES)
    if args.num_verses > 0:
        verse_set = list(popular_verses[:args.num_verses]['verse']) # process popular verses into list
    else:
        verse_set = list(popular_verses['verse'])
    verse_set = [VERSE_TO_TEXT[v] for v in verse_set if v in VERSE_TO_TEXT.keys()]
    
    sermon_df = pd.read_csv(args.sermon_path, header=None)
    
    sermon_df.columns = ['index', 'link', 'denomination',
                             'author', 'churchName', 'churchAddress',
                             'unknown1', 'unknown2', 'unknown3',
                             'date', 'title', 'verses_list',
                             'topicsList', 'ratingsNumber',
                             'rating', 'text']
    print(f"example verse list: {sermon_df['verses_list'].sample(5)}")
    
    sermon_df.set_index('index', inplace=True)
    if args.sample > -1:
        sermon_df = sermon_df.sample(args.sample)

    if args.clean_text:
        sermon_df['text'] = sermon_df['text'].apply(str).apply(clean)
        
    #citation_list = [] # should be sermon index to list of Reference objects
    chunk_size = args.chunk_size
    enum_sermons = list(enumerate(sermon_df['text']))
    enum_sermons = [(e[0], e[1], verse_list) for e, verse_list in zip(enum_sermons, sermon_df['verses_list'])]
    enum_sermon_chunks = [enum_sermons[i:i + chunk_size] for i in range(0, len(enum_sermons), chunk_size)]
    pool = Pool()
    
    # get citations!
    if not args.only_gold_verses:
        sermon_citation_chunks = pool.map(retrieve_references, enum_sermon_chunks)
        print("Finished mapping")
        citation_list = [e for sublist in sermon_citation_chunks for e in sublist]
        suffix = ''
    else:
        sermon_citation_chunks = pool.map(retrieve_whole_sermon_verse_labels, enum_sermon_chunks)
        print("Finished mapping")
        citation_list = [e for sublist in sermon_citation_chunks for e in sublist]
        suffix = '_onlygoldlabels'
    
    path = f'{args.output_dir}fsm{args.fsm_threshold}{suffix}'
    if args.num_verses > -1:
        path += '-' + str(args.num_verses)
    if args.sample > -1:
        path += '_sample' + str(args.sample)
        
    with open(f'{path}.json', 'w') as f:
        json.dump(citation_list, f)
        print(f"Dumped citation list to {path}.json")
