# how many speakers use "god"?
from data.data_utils import tokenizer
from data.congress_utils import load_full_df_from_raw
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

def find_kw(x, kw):
    return kw in tokenizer.tokenize(x)

def get_ngram_with_kw(x, kw, n=3):
    # get ngrams with keyword in them
    words = tokenizer.tokenize(x)

    appearances = []
    # find instances of keyword
    for i, word in enumerate(words):
        if word == kw:
            # get ngram
            start = max(0, i - n + 1)
            end = min(len(words), i + n)
            appearances.append('_'.join(words[start:end]))

    return appearances


# load data -- congress_df
df = load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True)
N = 5

kws = ["God", "faith"]
kw_to_ngrams = {kw: {} for kw in kws} # token --> ngram --> count

for kw in kws:
    print("Word:", kw)
    # find all speakers who use the keyword
    df_kw = df[df['text'].progress_apply(lambda x: find_kw(x, kw))]
    # get ngrams with keyword in them
    for idx, row in tqdm(df_kw.iterrows(), total=len(df_kw)):
        doc = row['text']
        appearances = get_ngram_with_kw(doc, kw, n=N)
        for appearance in appearances:
            if appearance not in kw_to_ngrams[kw].keys():
                kw_to_ngrams[kw][appearance] = 1
            else:
                kw_to_ngrams[kw][appearance] += 1


# save the ngram dict
import json
with open(f'/data/laviniad/congress_errata/God_ngram_dict.json', 'w') as f:
    json.dump(kw_to_ngrams["God"], f)

with open(f'/data/laviniad/congress_errata/faith_ngram_dict.json', 'w') as f:
    json.dump(kw_to_ngrams["faith"], f)
    
