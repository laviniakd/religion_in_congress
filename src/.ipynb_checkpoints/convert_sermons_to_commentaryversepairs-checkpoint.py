import pandas as pd
from data.data_utils import load_sermons, chunk_sermons
from data.data_utils import get_verse_dicts
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
TEMP_PATH = '/data/laviniad/sermons-ir/sermoncentral/fsm/train.csv'
OUT_PATH = '/data/laviniad/sermons-ir/commentary-verse-dfs/train/'
context_windows = [(1,1), (2,2), (3,3), (4,4),
                   (0,1), (1,0), (2,0), (0,2),
                   (0,3), (3,0), (4,0), (0,4)]


verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
sermon_df = pd.read_csv(TEMP_PATH)


for c in tqdm(context_windows):
    M = c[0]
    N = c[1]

    def chunk(df):
        return chunk_sermons(df, fuzzy_citation_dict, M=M, N=N, also_match_verses=(not only_match_citations))

    only_match_citations = True  # controls whether we also include quotations themselves

    print("Now chunking and converting to pairs")
    chunk_size = 2000
    chunks = [sermon_df[i:i + chunk_size] for i in range(0, len(sermon_df), chunk_size)]
    pool = Pool()
    print("Beginning parallelized conversion")
    results = pool.map(chunk, chunks)
    print("Done processing")
    chunked_dfs = pd.concat(results)

    OUT_STR_PATH = OUT_PATH + 'all_' + str(M) + 'before' + str(N) + 'after.csv'
    chunked_dfs.to_csv(OUT_STR_PATH)
    print("Wrote to " + OUT_STR_PATH)
