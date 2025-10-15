import pandas as pd
from data.congress_utils import load_full_df_from_raw
from tqdm import tqdm
import nltk

congress_path = "/data/corpora/congressional-record/"

# load keywords from full_list_new.txt

KWS = "src/keyword_list_construction/full_list_new.txt"

with open(KWS, 'r') as f:
    kws = f.readlines()
    kws = [kw.strip() for kw in kws]

# load data

wcs = {}
wcs = {kw: 0 for kw in kws}

congress_df = load_full_df_from_raw(congress_path, remove_procedural_speeches=True)
for idx, row in tqdm(congress_df.iterrows()):
    text = row['text']
    tokens = nltk.word_tokenize(text)

    for kw in kws:
        wcs[kw] += tokens.count(kw)

wcs_df = pd.DataFrame(wcs.items(), columns=['keyword', 'count'])
wcs_df.to_csv("data/wordcounts/wordcounts.csv")

