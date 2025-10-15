import pandas as pd
import os
import nltk
import re
import argparse
from tqdm import tqdm
import string
from nltk.tokenize import word_tokenize
from data import presidential_utils, coca_utils, congress_utils
from data.load_reference_targets_from_original_df import clean, return_cleaning_string_patterns
from collections import Counter

tqdm.pandas()

COCA_PATH = '/data/laviniad/COCA/'
PRESIDENTIAL_PATH = '/data/laviniad/presidential/'
CONGRESSIONAL_PATH = '/data/corpora/congressional-record/'
SERMON_PATH = '/data/laviniad/sermons-ir/sermoncentral/with_columns.csv'

COMPUTE_COCA = False
COMPUTE_PRES = False
COMPUTE_SERMONS = True
COMPUTE_CONGRESSIONAL = True


parser = argparse.ArgumentParser()
parser.add_argument('--prep_tokens', action='store_true')
parser.add_argument('--min_token_length', default=3, type=int)
# args for coca, presidential, congressional, sermons, output dir
parser.add_argument('--coca_dir', default=COCA_PATH, type=str)
parser.add_argument('--presidential_dir', default=PRESIDENTIAL_PATH, type=str)
parser.add_argument('--congressional_dir', default=CONGRESSIONAL_PATH, type=str)
parser.add_argument('--sermons_dir', default=SERMON_PATH, type=str)

parser.add_argument('--output', default='/data/laviniad/sermons-ir/token-frequencies/', type=str)
args = parser.parse_args()

# set args as constants :|
OUTPUT = args.output
COCA_PATH = args.coca_dir
PRESIDENTIAL_PATH = args.presidential_dir
CONGRESSIONAL_PATH = args.congressional_dir
SERMON_PATH = args.sermons_dir

VERSE_REGEX, QUOTE_PATTERN, LINK_PATTERN, HTML_PATTERN = return_cleaning_string_patterns()


def replace_numbers_and_punct(t):
        t = clean(t)
        t = re.sub(r'\d+', 'NUM', t)
        t = re.sub(r'[^\w\s]', '', t) # remove punctuation
        return t


def get_freq_df(df, prep_tokens=False, minimum_length=3):

    if prep_tokens:
        tokens = [replace_numbers_and_punct(token).strip() for tokens in tqdm(df['text']) for token in tokens if (len(token) >= minimum_length) and (token.isalnum())]
    else:
        tokens = [token.strip() for tokens in df['text'] for token in tokens]
    #bigrams = [[' '.join(tokens[i:i+1]) for i,t in enumerate(tokens) if (i < len(tokens) - 1)] for tokens in df['text']]
    #bigrams = [bigram for b in bigrams for bigram in b]
    
    token_freq = dict(Counter(tokens))
    #bigram_freq = dict(Counter(bigrams))
    freq_df = pd.DataFrame(token_freq, index=[0]).T
    print(freq_df.head())
    #bigram_freq_df = pd.DataFrame(bigram_freq, index=[0]).T
    return freq_df#, bigram_freq_df
    

## COCA
if COMPUTE_COCA:
    print("Computing COCA...")
    coca_df = coca_utils.load_COCA_from_raw(COCA_PATH)
    print(coca_df.head())   
    coca_df['text'] = coca_df['text'].progress_apply(word_tokenize)
    print(f"Loaded COCA dataframe with {len(coca_df.index)} rows...")

    coca_freq_df = get_freq_df(coca_df, 
                                                    prep_tokens=args.prep_tokens, 
                                                    minimum_length=args.min_token_length
                                                   )
    coca_freq_df.to_csv(OUTPUT + 'coca.csv')
    #coca_bigram_freq_df.to_csv(OUTPUT + 'coca_bigram.csv')

## PRESIDENTIAL
if COMPUTE_PRES:
    print("Computing Presidential speeches etc...")
    pres_dict = presidential_utils.load_presidential_from_raw(PRESIDENTIAL_PATH)
    print("Loading DF...")
    pres_df = presidential_utils.output_comprehensive_df(pres_dict)
    pres_df['text'] = pres_df['text'].progress_apply(word_tokenize)
    print(f"Loaded presidential dataframe with {len(pres_df.index)} rows...")

    presidential_token_freq = get_freq_df(pres_df,
                                          prep_tokens=args.prep_tokens,
                                          minimum_length=args.min_token_length)
    presidential_token_freq.to_csv(OUTPUT + 'presidential.csv')
    #presidential_token_freq.to_csv(OUTPUT + 'presidential_bigram.csv')
    
if COMPUTE_CONGRESSIONAL:
    print("Computing Congressional record...")
    congress_df = congress_utils.load_full_df_from_raw(CONGRESSIONAL_PATH, remove_procedural_speeches=True, nonprocedural_indices_path="/data/laviniad/congress_errata/nonprocedural_indices.json")
    congress_df['text'] = congress_df['text'].progress_apply(word_tokenize)
    print(f"Loaded congressional dataframe with {len(congress_df.index)} rows...")

    congress_token_freq = get_freq_df(congress_df,
                                      prep_tokens=args.prep_tokens,
                                      minimum_length=args.min_token_length)
    congress_token_freq.to_csv(OUTPUT + 'congress.csv')

## SERMONS
if COMPUTE_SERMONS:
    print("Computing sermons...")
    sermon_df = pd.read_csv(SERMON_PATH)
    sermon_df['text'] = sermon_df['text'].apply(str).progress_apply(word_tokenize)
    print(f"Loaded sermon dataframe with {len(sermon_df.index)} rows...")
    
    sermon_token_freq = get_freq_df(sermon_df,
                                                        prep_tokens=args.prep_tokens, 
                                                        minimum_length=args.min_token_length
                                                        )
    sermon_token_freq.to_csv(OUTPUT + 'sermons.csv')
    #sermon_token_freq.to_csv(OUTPUT + 'sermons_bigram.csv')
