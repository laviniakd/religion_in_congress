import pandas as pd
from data.data_utils import load_spacy_sentencizer, load_sermons, get_verses_from_sermon_without_context
from data.data_utils import get_verse_dicts
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import repeat

FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
TEMP_PATH = '/data/laviniad/sermons-ir/sermoncentral/fsm/train.csv'
OUT_PATH = '/data/laviniad/sermons-ir/whole-sermon-dfs/fsm/train.csv'


## spacy sentence tokenizer 
nlp = load_spacy_sentencizer()
assert(nlp is not None)

verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
sermon_df = pd.read_csv(TEMP_PATH)


def individual_row_verses(row, nlp):
    sermon_text = str(row.text)
    
    docs = nlp(str(sermon_text), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    sent_toked_sermon = [t.text for t in docs.sents]

    verse_list = get_verses_from_sermon_without_context(sent_toked_sermon, fuzzy_citation_dict)
    row['verses'] = [v['verse'] for v in verse_list]
    return row


def process_chunk(sermon_df_chunk, nlp, id=None):
    results = []
    
    sermon_df_chunk = sermon_df_chunk.apply(lambda x: individual_row_verses(x, nlp), axis=1)
    return sermon_df_chunk


only_match_citations = True  # controls whether we also include quotations themselves

print("Now chunking and converting to pairs")
chunk_size = 2000
chunks = [sermon_df[i:i + chunk_size] for i in range(0, len(sermon_df), chunk_size)]
pool = Pool()
print("Beginning parallelized conversion")
results = pool.starmap(process_chunk, zip(chunks, repeat(nlp)))
print("Done processing")
chunked_dfs = pd.concat(results)

chunked_dfs.to_csv(OUT_PATH)
print("Wrote to " + OUT_PATH)
