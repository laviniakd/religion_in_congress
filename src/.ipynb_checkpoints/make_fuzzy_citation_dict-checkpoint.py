import json
import pandas as pd

## TEMPORARILY SWITCHING TO FUZZYWUZZY
# from rapidfuzz import fuzz
# from rapidfuzz.process import extractOne

from fuzzywuzzy.process import extractOne

from tqdm import tqdm
from data.data_utils import return_verses_in_passage
from data.bible_utils import bible_helper

# want to produce dict that sends all permutations of verse to canonical KJV version --
# e.g. Heb. 1:2, He 1:2, Hebrews 1:2 --> Hebrews 1:2
fuzzy_citation_dict = {}
unmatched_list = []
OUTPUT_PATH = '/shared/3/projects/sermons-ir/fuzzy_citation_dict.json'
TEMP_PATH = '/shared/3/projects/sermons-ir/sermoncentral/with_columns.csv'
LIST_PATH = '/shared/3/projects/sermons-ir/sermoncentral/unmatched_but_regexed.txt'

sermon_df = pd.read_csv(TEMP_PATH)
sermon_df['recognized_verses'] = sermon_df['text'].apply(str).apply(lambda x: return_verses_in_passage(x, True))
recognized_verses = []
for verse_list in tqdm(sermon_df['recognized_verses']):
    for s in list(verse_list):
        if s not in recognized_verses:
            recognized_verses.append(s)

print("Recognized verses preview: ", str(recognized_verses[:10]))

BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
bible_df = bible_helper(BIBLE_PATH)
books = list(set(bible_df['book']))
print("Books: ", str(books))

THRESHOLD = 90

# iterate through verses and attempt to match
for v in tqdm(recognized_verses):
    split_v = v.split()

    # handle 'number name' book format
    if split_v[0].isnumeric():
        book = split_v[0] + ' ' + split_v[1]
        has_num = True
        rest = split_v[2:]
    else:
        book = split_v[0]
        has_num = False
        rest = split_v[1:]

    matched = False

    # reduce books to match on -- abbreviations still have same first letter
    limited_book_set = [b for b in books if b.lower()[0] == book.lower()[0]]

    if len(limited_book_set) != 0:
        # first pass is FSM
        chosen_b = extractOne(book.lower(), [b.lower() for b in books])[0]

        if has_num:  # chapters match
            if book[0] == chosen_b[0]:
                fuzzy_citation_dict[v] = ' '.join([chosen_b] + rest)  # b is the canonical verse format

                matched = True
        else:
            fuzzy_citation_dict[v] = ' '.join([chosen_b] + rest)  # b is the canonical verse format

            matched = True

        # second is bc not using partial ratio
        for b in books:
            if not matched:
                print("Still have not found match for verse ", v)
                if has_num and b.startswith(' '.join(split_v[:2])): # number + book
                    fuzzy_citation_dict[v] = ' '.join([b] + rest)
                    matched = True

                elif b.startswith(split_v[0]):
                    fuzzy_citation_dict[v] = ' '.join([b] + rest)
                    matched = True

    if not matched:
        print("Found NO match for verse: " + str(v))
        unmatched_list.append(v)

print("Fuzzy citation dict produced!")
with open(OUTPUT_PATH, 'w') as f:
    json.dump(fuzzy_citation_dict, f)

with open(LIST_PATH, 'w') as f:
    f.writelines(unmatched_list)
