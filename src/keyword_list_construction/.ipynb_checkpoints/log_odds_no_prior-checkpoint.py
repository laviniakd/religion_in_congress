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
SC_END = 'sermon_coca_odds.json'
CS_END = 'coca_sermon_odds.json'
SR_END = 'sermon_congress_odds.json'
RS_END = 'congress_sermon_odds.json'

parser = argparse.ArgumentParser()
parser.add_argument('--prep_tokens', action='store_true')
parser.add_argument('--min_count', default=5, type=int)
parser.add_argument('--too_common_filter', default=100, type=int, 
                    help="Integer describing the frequency rank of a token in COCA at which it is removed from the keyword list")
parser.add_argument('--wrt_congress', action='store_true')
args = parser.parse_args()


def prep_token(t):
    t = t.lower().replace('-', ' ')
    return t

if args.prep_tokens:
    STOP_WORDS = stopwords.words('english')
    bible_metadata = pd.read_csv('/data/laviniad/bible-metadata/CSV/People.csv')
    NAMES = set(bible_metadata['name'].apply(lambda x: x.lower()))
    #OUTPUT_PATH += 'filtered/'

print("Loading token frequencies...")

count_removed = 0
if args.wrt_congress:
    with open(TOKEN_FREQS + 'congress.csv') as csvfile:
        drc = csv.DictReader(csvfile, fieldnames=['token', 'count'])
        coca_token_freq = {}
        
        for r in drc:
            prepped_token = prep_token(r['token'])
            if prepped_token not in coca_token_freq.keys():
                coca_token_freq[prepped_token] = 0
                
            coca_token_freq[prepped_token] += int(r['count'])
else:
    with open(TOKEN_FREQS + 'coca.csv') as csvfile:
        drc = csv.DictReader(csvfile, fieldnames=['token', 'count'])
        coca_token_freq = {prep_token(r['token']): int(r['count']) for r in drc}

with open(TOKEN_FREQS + 'sermons.csv') as csvfile:
    drs = csv.DictReader(csvfile, fieldnames=['token', 'count'])
    sermon_token_freq = {}
    
    for r in drs:
        prepped_token = prep_token(r['token'])
        if prepped_token not in sermon_token_freq.keys():
            sermon_token_freq[prepped_token] = 0
                
        sermon_token_freq[prepped_token] += int(r['count'])
        
print("Files loaded!")

coca_tokens = list(coca_token_freq.keys())
sermon_tokens = list(sermon_token_freq.keys())
print(f'{len(coca_tokens)} C tokens')
print(f'{len(sermon_tokens)} sermon tokens')

FILTER_LIMIT = args.too_common_filter
POP_WORDS = list(coca_token_freq.items())
POP_WORDS.sort(key=lambda x: x[1])
POP_WORDS = [e[0] for e in POP_WORDS[:FILTER_LIMIT]]


def not_too_common(keyword):
    return not (keyword in POP_WORDS)


def is_verb(keyword):
    return any(syn.pos() == 'v' for syn in wn.synsets(keyword))


if args.prep_tokens:
    print("Removing tokens according to stop word list, verb exclusion, frequency criterion, and name filter")
    original_coca_length = len(coca_tokens)
    original_sermon_length = len(sermon_tokens)
    coca_tokens = [t for t in coca_tokens if coca_token_freq[t] > args.min_count]
    sermon_tokens = [t for t in sermon_tokens if sermon_token_freq[t] > args.min_count]
    print(f'Removed {original_coca_length - len(coca_tokens)} tokens in C corpus and {original_sermon_length - len(sermon_tokens)} in sermons due to low frequency')
    
    original_coca_length = len(coca_tokens)
    original_sermon_length = len(sermon_tokens)
    coca_tokens = [t for t in coca_tokens if t not in STOP_WORDS]
    sermon_tokens = [t for t in sermon_tokens if t not in STOP_WORDS]
    print(f'Removed {original_coca_length - len(coca_tokens)} tokens in C corpus and {original_sermon_length - len(sermon_tokens)} in sermons due to being stop words')
    
    original_coca_length = len(coca_tokens)
    original_sermon_length = len(sermon_tokens)   
    coca_tokens = [t for t in tqdm(coca_tokens) if (not is_verb(t) and (t in sermon_tokens))]
    sermon_tokens = [t for t in tqdm(sermon_tokens) if (not is_verb(t) and (t in coca_tokens))]
    print(f'Removed {original_coca_length - len(coca_tokens)} tokens in C corpus and {original_sermon_length - len(sermon_tokens)} in sermons due to being verbs (or not in the other corpus)')
    
    original_coca_length = len(coca_tokens)
    original_sermon_length = len(sermon_tokens)  
    coca_tokens = [t for t in tqdm(coca_tokens) if (not_too_common(t) and not (t in NAMES))]
    sermon_tokens = [t for t in tqdm(sermon_tokens) if (not_too_common(t) and not (t in NAMES))]
    print(f'Removed {original_coca_length - len(coca_tokens)} tokens in C corpus and {original_sermon_length - len(sermon_tokens)} in sermons due to not being too common')
        
    

print(f"Number of COCA tokens: {len(coca_tokens)}")
print(f"Number of sermon tokens: {len(sermon_tokens)}")

sc_log_odds = {}
cs_log_odds = {}
# all smoothed
print("Computing log odds...")

for token in tqdm(set(coca_tokens + sermon_tokens)):
    scount = 0
    if token in sermon_token_freq.keys():
        scount = sermon_token_freq[token]
    ccount = 0
    if token in coca_token_freq.keys():
        ccount = coca_token_freq[token]
    
    p1 = np.log(scount + 1) / (len(sermon_token_freq) + 2)
    p2 = np.log(ccount + 1) / (len(coca_token_freq) + 2)
    sc_log_odds[token] = p1 - p2
    cs_log_odds[token] = p2 - p1

sc_log_odds = dict(sorted(sc_log_odds.items(), key=lambda x:x[1]))
cs_log_odds = dict(sorted(cs_log_odds.items(), key=lambda x:x[1]))

if not args.wrt_congress:
    print("Dumping log odds...")
    with open(OUTPUT_PATH + SC_END, 'w') as f:
        json.dump(sc_log_odds, f)

    with open(OUTPUT_PATH + CS_END, 'w') as f:
        json.dump(cs_log_odds, f)
else:
    print("Dumping log odds...")
    with open(OUTPUT_PATH + SR_END, 'w') as f:
        json.dump(sc_log_odds, f)

    with open(OUTPUT_PATH + RS_END, 'w') as f:
        json.dump(cs_log_odds, f)
