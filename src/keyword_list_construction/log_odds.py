import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import csv
import argparse
#from data.data_utils import load_stop_words
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
nltk.download('universal_tagset')
nltk.download('stopwords')

TOKEN_FREQS = '/data/laviniad/sermons-ir/token-frequencies/'
OUTPUT_PATH = '/data/laviniad/sermons-ir/log_odds/final/'
SR_END = 'sermon_congress_odds_FINAL.json'
RS_END = 'congress_sermon_odds_FINAL.json'

parser = argparse.ArgumentParser()
parser.add_argument('--prep_tokens', action='store_true')
parser.add_argument('--min_count', default=5, type=int)
parser.add_argument('--too_common_filter', default=100, type=int, 
                    help="Integer describing the frequency rank of a token at which it is removed from the keyword list")

parser.add_argument('--output', default=OUTPUT_PATH, type=str)
parser.add_argument('--token_freq_dir', default=TOKEN_FREQS, type=str)
parser.add_argument('--alpha', default=0.1, type=float)

args = parser.parse_args()

ALPHA = args.alpha

def prep_token(t):
    t = t.replace('-', ' ')
    return t

if args.prep_tokens:
    STOP_WORDS = stopwords.words('english')
    bible_metadata = pd.read_csv('/data/laviniad/bible-metadata/CSV/People.csv')
    NAMES = set(bible_metadata['name'].apply(lambda x: x.lower()))
    #OUTPUT_PATH += 'filtered/'

print("Loading token frequencies...")

count_removed = 0

with open(TOKEN_FREQS + 'congress.csv') as csvfile:
    drc = csv.DictReader(csvfile, fieldnames=['token', 'count'])
    congress_token_freq = {}
        
    for r in drc:
        prepped_token = prep_token(r['token'])
        if prepped_token not in congress_token_freq.keys():
            congress_token_freq[prepped_token] = 0
                
        congress_token_freq[prepped_token] += int(r['count'])

with open(TOKEN_FREQS + 'coca.csv') as csvfile:
    drc = csv.DictReader(csvfile, fieldnames=['token', 'count'])
    coca_token_freq = {prep_token(r['token']): int(r['count']) for r in drc}
    for r in drc:
        prepped_token = prep_token(r['token'])
        if prepped_token not in coca_token_freq.keys():
            coca_token_freq[prepped_token] = 0
                
        coca_token_freq[prepped_token] += int(r['count'])

with open(TOKEN_FREQS + 'sermons.csv') as csvfile:
    drs = csv.DictReader(csvfile, fieldnames=['token', 'count'])
    sermon_token_freq = {}
    
    for r in drs:
        prepped_token = prep_token(r['token'])
        if prepped_token not in sermon_token_freq.keys():
            sermon_token_freq[prepped_token] = 0
                
        sermon_token_freq[prepped_token] += int(r['count'])
        
print("Files loaded!")

congress_tokens = list(congress_token_freq.keys())
sermon_tokens = list(sermon_token_freq.keys())
print(f'{len(congress_tokens)} C tokens')
print(f'{len(sermon_tokens)} sermon tokens')

FILTER_LIMIT = args.too_common_filter
POP_WORDS = list(congress_token_freq.items())
#for p in POP_WORDS:
    # if p is capitalized
    #if len(p) > 0 and p[0].isupper():
    #    print(p)

POP_WORDS.sort(key=lambda x: x[1])
POP_WORDS = [e[0] for e in POP_WORDS[:FILTER_LIMIT]]


def not_too_common(keyword):
    return not (keyword in POP_WORDS)


def is_verb(keyword):
    return any(syn.pos() == 'v' for syn in wn.synsets(keyword))


if args.prep_tokens:
    print("Removing tokens according to stop word list, verb exclusion, frequency criterion, and name filter")
    
    def filter_tokens(tokens, freq_dict, min_count, stop_words, names, other_tokens):
        original_length = len(tokens)
        tokens = [t for t in tqdm(tokens) if freq_dict[t] > min_count]
        print(f'Removed {original_length - len(tokens)} tokens due to low frequency')
        
        original_length = len(tokens)
        tokens = [t for t in tqdm(tokens) if t not in stop_words]
        print(f'Removed {original_length - len(tokens)} tokens due to being stop words')
        
        original_length = len(tokens)
        tokens = [t for t in tqdm(tokens) if (not is_verb(t) and (t in other_tokens))]
        print(f'Removed {original_length - len(tokens)} tokens due to being verbs (or not in the other corpus)')
        
        original_length = len(tokens)
        tokens = [t for t in tqdm(tokens) if (not_too_common(t) and not (t in names))]
        print(f'Removed {original_length - len(tokens)} tokens due to not being too common or being names')

        original_length = len(tokens)
        tokens = [t for t in tqdm(tokens) if 'NUM' not in t]
        print(f'Removed {original_length - len(tokens)} tokens due to including numbers')
        
        return tokens
    
    STOP_WORDS = stopwords.words('english')
    bible_metadata = pd.read_csv('/data/laviniad/bible-metadata/CSV/People.csv')
    NAMES = set(bible_metadata['name'].apply(lambda x: x.lower()))
    
    congress_tokens = filter_tokens(congress_tokens, congress_token_freq, args.min_count, STOP_WORDS, NAMES, sermon_tokens)
    sermon_tokens = filter_tokens(sermon_tokens, sermon_token_freq, args.min_count, STOP_WORDS, NAMES, congress_tokens)
    

print(f"Number of congress tokens: {len(congress_tokens)}")
print(f"Number of sermon tokens: {len(sermon_tokens)}")

sc_log_odds = {}
cs_log_odds = {}
# all smoothed
print("Computing log odds...")

# how many words are in each corpus
num_sermon_words = sum(sermon_token_freq.values())
num_congress_words = sum(congress_token_freq.values())
num_coca_words = sum(coca_token_freq.values())
word_not_in_coca = []

a_0 = ALPHA * num_coca_words
full_vocab = set(congress_tokens + sermon_tokens)
for token in tqdm(full_vocab):
    scount = 0
    if token in sermon_token_freq.keys():
        scount = sermon_token_freq[token] # is f^i_w in jurafsky

    ccount = 0
    if token in congress_token_freq.keys():
        ccount = congress_token_freq[token] # is f^j_w in jurafsky

    
    if token in coca_token_freq.keys(): # add smoothing
        scount += ALPHA * coca_token_freq[token] # scount is now f^i_w + a * b_w
        ccount += ALPHA * coca_token_freq[token] # ccount is now f^j_w + a * b_w
        nsw_new = num_sermon_words + a_0 # this is n^i + a_0
        ncw_new = num_congress_words + a_0 # this is n^j + a_0
    else:
        word_not_in_coca.append(token)
    
    p1 = np.log(scount) - np.log(nsw_new - scount)
    p2 = np.log(ccount) - np.log(ncw_new - ccount)
    #var = (1 / (scount)) + (1 / (ccount))

    sc_log_odds[token] = (p1 - p2) #/ np.sqrt(var)
    cs_log_odds[token] = (p2 - p1) #/ np.sqrt(var)

sc_log_odds = dict(sorted(sc_log_odds.items(), key=lambda x:x[1]))
cs_log_odds = dict(sorted(cs_log_odds.items(), key=lambda x:x[1]))

print("Number of words not in COCA: ", len(word_not_in_coca))

print("Dumping log odds...")
with open(OUTPUT_PATH + SR_END, 'w') as f:
    json.dump(sc_log_odds, f)

with open(OUTPUT_PATH + RS_END, 'w') as f:
    json.dump(cs_log_odds, f)
