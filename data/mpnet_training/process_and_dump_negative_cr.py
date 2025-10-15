import pandas as pd
import random
from random import choices
import re
from tqdm import tqdm
import nltk
from itertools import product
from data import presidential_utils, congress_utils
from data.congress_utils import induce_party_and_state
from data.bible_utils import comp_bible_helper


# loading data
print("Loading congressional data and inducing party")
congress_df = pd.read_csv('/data/laviniad/congress_errata/congress_df.csv') # already has nonprocedural filtered out
congress_df = induce_party_and_state(congress_df)
assert('party' in congress_df.columns)

print("Loading bible verses")
TAG_RE = re.compile(r'<[^>]+>')
MIN_LENGTH = 5 # sentences with less than 5 whitespace tokens are not useful -- also doing length limiting

def remove_tags(text):
    return TAG_RE.sub('', text)

bible_df = comp_bible_helper()
pop_verses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv')
n = 1000 # VERY generous
# remove the first verse, which is 'UNKNOWN' (artifact of sermon data)
pop_citations = list(pop_verses['verse'].iloc[1:n+2])

bible_df['King James Bible'] = bible_df['King James Bible'].apply(remove_tags)
bible_df['Verse'] = bible_df['Verse'].apply(lambda x: x.lower())
limited_bible_df = bible_df[bible_df['Verse'].apply(lambda x: x in pop_citations)]
limited_verses = limited_bible_df['King James Bible']
limited_verse_to_citation = dict(zip(limited_verses, limited_bible_df['Verse']))
limited_citation_to_verse = {v: k for k,v in limited_verse_to_citation.items()}

# equally sampling from each party + year; creating pairs
print("Creating combinations")
NUM_EXAMPLES = 100000
negative_examples = []
years = list(range(max(1990, congress_df.year.min()), min(2024, congress_df.year.max() + 1)))
parties = ['Democrat', 'Republican']

combinations = list(product(years, parties))
num_examples_from_each_pair = int(NUM_EXAMPLES / len(combinations))

def extract_random_sentence(text):
    sentences = [s for s in nltk.sent_tokenize(text) if len(s.split()) >= MIN_LENGTH]
    if len(sentences) == 0:
        return ''
    sent_choice = random.choice(sentences)
    return sent_choice
    

print(f"Starting stratified sampling with {len(combinations)} combinations")
sampled_total = 0
for c in combinations:
    congress_df_limited = congress_df[(congress_df['year'] == c[0]) & (congress_df['party'] == c[1])]
    print(f"Length of congress_df_limited is {len(congress_df_limited.index)} with party {c[1]} and year {c[0]}")
    if len(congress_df_limited.index) > 0:
        sampled = congress_df_limited.sample(min(num_examples_from_each_pair, len(congress_df_limited.index)))
        sampled['text'] = sampled['text'].apply(extract_random_sentence) # instead of taking whole document, just take one sentence
        sampled = sampled[sampled['text'] != ''] # remove empty strings (reflecting lack of suitable sentences, i.e. with more length)
        sampled_total += len(sampled.index)
        sampled['random_verse'] = choices(list(limited_verses), k=len(sampled.index))
        sampled['citation'] = sampled['random_verse'].apply(lambda x: limited_verse_to_citation[x])
        negative_examples.append(sampled)
    else:
        print("Not sampling due to no examples of pair")
    
negative_df = pd.concat(negative_examples)

assert(len(negative_df.index) == sampled_total) # ensure the concatenation behaved as expected
negative_df.to_csv('/data/laviniad/congress_errata/negative_examples_df.csv')
print("Dumped to /data/laviniad/congress_errata/negative_examples_df.csv")
