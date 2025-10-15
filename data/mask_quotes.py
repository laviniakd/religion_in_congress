import pandas as pd
from tqdm import tqdm
import json
from data.data_utils import return_verses_in_passage
from data.data_utils import mask_quotes_using_fsm
from data.bible_utils import bible_helper
from data.data_utils import get_verse_dicts
from multiprocessing import Pool, cpu_count

import random

# masks quotes using the fuzzy strategy and outputs new df with
# 1) recognized verses in column
# 2) text with rectified verses (i.e., similar canonical verses per FSM)
# 3) masked version

tqdm.pandas()
FUZZY_CITATION_DICT_PATH = '/shared/3/projects/sermons-ir/fuzzy_citation_dict.json'
FULL_SERMON_PATH = '/shared/3/projects/sermons-ir/sermoncentral/with_columns.csv'
OUTPUT_PATH = '/shared/3/projects/sermons-ir/sermoncentral/fsm_with_columns.csv'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'

THRESHOLD = 90  # FSM score required to declare sentences matching


verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)

with open('/shared/3/projects/sermons-ir/verse_to_text.json', 'w') as f:
    json.dump(verse_text_dict, f)

#print("Preview of verse-text dict", str(verse_text_dict.items()[-40:-20]))

def debug_printing():
    print("DEBUGGING BY PRINTING OUT FUZZY CITATION DICT AND VERSE TEXT DICT: ")
    print("Fuzzy citation dict: ")
    print(list(fuzzy_citation_dict.items())[:50])
    print("Verse-text dict: ")
    print(list(verse_text_dict.items())[:50])
    print()
    print("Now iterating through fuzzy citation dict to ensure things worked out")
    print()
    for noncanon, v in fuzzy_citation_dict.items():
        if noncanon != v:
            print(noncanon)
            print(v)


#debug_printing()

sermon_df = pd.read_csv(FULL_SERMON_PATH)

##### DEBUGGING
# sermon_df = sermon_df.sample(100)

sermon_df['text'] = sermon_df['text'].apply(str)
print("Successfully cast 'text' column")

sermon_df['recognized_verses'] = sermon_df['text'].progress_apply(
    lambda x: return_verses_in_passage(x, split_ranges=True)
)
print("Successfully labeled recognized verses")

# sermon_df = sermon_df.progress_apply(lambda row: mask_quotes_using_fsm(
#     row,
#     verse_text_dict,
#     fuzzy_citation_dict,
#     fuzz_threshold=THRESHOLD
# ), axis=1)


def mask(rows):
    return rows.progress_apply(lambda row: mask_quotes_using_fsm(
        row,
        verse_text_dict,
        fuzzy_citation_dict,
        fuzz_threshold=THRESHOLD), axis=1)


chunk_size = 2000
chunks = [sermon_df[i:i+chunk_size] for i in range(0, len(sermon_df), chunk_size)]
pool = Pool()
print("Beginning parallelized masking")
results = pool.map(mask, chunks)
print("Done processing")

# Concatenate the results into a single DataFrame
sermon_df = pd.concat(results)

print("Successfully output corrected and masked versions of sermon")

sermon_df.to_csv(OUTPUT_PATH)
print("output file")
