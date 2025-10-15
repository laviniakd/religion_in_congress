#!/usr/bin/env python
import pandas as pd
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from collections import Counter
from data import presidential_utils, congress_utils
from data.congress_utils import induce_party_and_state
from data.bible_utils import comp_bible_helper
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from rapidfuzz import process, fuzz
from tqdm.notebook import tqdm

DEBUG = False
FSM_LIMIT = 65
LOAD_CSV = False
ONLY_BIBLE_EXAMPLES = False
VERSE_REGEX = r"\b(?:[1-3]?\s?[A-Za-z]+(?:\s?[1-3]?[0-9]))(?::\s?[1-9][0-9]*(?:-[1-9][0-9]*)?)?\b"
BIBLE_LIST = ['King James Bible','American Standard Version', 'American King James Version', 'Young\'s Literal Translation', 
              'World English Bible', 'Darby Bible Translation', 'Douay-Rheims Bible', 'Webster Bible Translation']

# load congressional data
CONGRESS_ERRATA_PATH = '/data/laviniad/congress_errata/'
if not ONLY_BIBLE_EXAMPLES:
    congress_df = congress_utils.load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True)
    congress_df = induce_party_and_state(congress_df)
    print("Loaded congressional dataframe")

# load popular bible verses
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

bible_df = comp_bible_helper()
pop_verses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv')
n = 500 # VERY generous
pop_citations = list(pop_verses['verse'].iloc[1:n+1]) # remove 'UNKNOWN', which is most popular (artifact of sermon data)

bible_df['King James Bible'] = bible_df['King James Bible'].apply(remove_tags)
bible_df['Verse'] = bible_df['Verse'].apply(lambda x: x.lower())
limited_bible_df = bible_df[bible_df['Verse'].apply(lambda x: x in pop_citations)]
limited_verses = limited_bible_df['King James Bible']
limited_verse_to_citation = dict(zip(limited_verses, limited_bible_df['Verse']))
limited_citation_to_verse = {v: k for k,v in limited_verse_to_citation.items()}

# define synonyms + FSM procedure
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().replace('_', ' '))
    return synonyms

bible_synonyms = set(get_synonyms("Bible"))
bible_pattern = "|".join(bible_synonyms) + "|" + "Old Testament" + "|" + "New Testament"

says_synonyms = set(get_synonyms("says"))
says_pattern = "|".join(says_synonyms)

def fuzzy_match_with_verse(text):
    result = process.extractOne(text, limited_verses, scorer=fuzz.token_sort_ratio) # example return: ('Dallas Cowboys', 83.07692307692308, 3)
    matched_verse, score = result[0], result[1]
    if score > FSM_LIMIT: # acceptable score
        return matched_verse, limited_verse_to_citation[matched_verse], score
    else:
        return '', 'no_quote', score

# find examples
if not ONLY_BIBLE_EXAMPLES:
    context_length = 80

    if DEBUG:
        congress_df = congress_df.sample(50000)

    if not LOAD_CSV:
        cong_examples = []
        final_pattern = rf'(?:{bible_pattern}) (?:{says_pattern})'
        for idx,row in congress_df.iterrows():
            text = row['text']
            assert(isinstance(text, str))
            sentences = sent_tokenize(text)

            for t in sentences:
                matches = re.findall(final_pattern, t)
                temp = []
                for match in matches:
                    matched_verse, verse_text_number, fsm_score = fuzzy_match_with_verse(t)

                    verse_citation = re.search(VERSE_REGEX, t)
                    if verse_citation:
                        verse_citation = verse_citation.group(0)
                    else:
                        verse_citation = 'no_citation'

                    temp.append({'text': t, 'speaker': row['speaker'], 'year': row['year'], 'matched_text': match,
                                'regex_citation': verse_citation, 'verse_quoted': verse_text_number, 
                                'verse_text': matched_verse, 'fsm_score_of_quote': fsm_score,
                                'party': row['party'], 'state': row['state']})

                cong_examples += temp

            if idx % 100000 == 0:
                print(f"On index {idx} of {len(congress_df.index)}")

        cong_examples = pd.DataFrame(cong_examples)
        cong_examples.to_csv(CONGRESS_ERRATA_PATH + 'positive_temp_df.csv')
    else: # do load csv
        cong_examples = pd.read_csv(CONGRESS_ERRATA_PATH + 'positive_temp_df.csv')

    print("Examples of fuzzy string matching: ")
    limited = cong_examples[cong_examples['verse_quoted'] != 'no_quote']
    limited_sample = limited.sample(min(len(limited.index), 50))
    for idx,row in limited_sample.iterrows():
        print("-" * 100)
        print(f"In this row, the string \"{row['text']}\" is matched with {row['verse_quoted']}, which has the text \"{limited_citation_to_verse[row['verse_quoted']]}\"")
        print(f"The score is {row['fsm_score_of_quote']}")

    if not DEBUG:
        cong_examples.to_csv(CONGRESS_ERRATA_PATH + 'positive_examples_df.csv')
        print(f"Dumped to {CONGRESS_ERRATA_PATH + 'positive_examples_df.csv'}")

# collect bible verse parallel examples
pop_citations = [i.lower() for i in pop_citations] # maintain casing
print(f"pop_citations: {pop_citations[:10]}")
parallel_bible_examples = []
for idx,row in bible_df.iterrows():
    if row['citation'].lower() in pop_citations:
        verse_versions = []
        for r in BIBLE_LIST:
            t = row[r]
            if isinstance(t, str):
                verse_versions.append(remove_tags(t))
        verse_versions = list(set(verse_versions))
    
        parallel_bible_examples.append({
            'verse_versions': verse_versions,
            'citation': row['citation'],
            'book': row['book'],
            'chapter': row['chapter'],
            'verse': row['verse']
        })
parallel_bible_examples = pd.DataFrame(parallel_bible_examples)
print(f"Have {len(parallel_bible_examples.index)} examples from parallel Bible versions")
    
    
if not DEBUG:
    parallel_bible_examples.to_csv(CONGRESS_ERRATA_PATH + 'parallel_positive_examples_df.csv')
    print(f"Dumped to {CONGRESS_ERRATA_PATH + 'parallel_positive_examples_df.csv'}")
    

if DEBUG:
    print("debugging done")
