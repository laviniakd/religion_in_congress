from data.data_utils import get_full_keywords
from data.congress_utils import load_full_df_from_raw, induce_party_and_state
from nltk import word_tokenize
from tqdm import tqdm
import json

N = 3

print("Loading keywords")
keywords = get_full_keywords(ONLY_GOD_TALK=True)
keywords = [k.lower() for k in keywords] # frankly, no point in not doing this given caps won't make a difference

print("Loading congressional dataframe")
congress_df = load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True)

ngram_dict = {k: {} for k in keywords} # token --> ngram --> count

def get_ngrams(input, n=3):
    return zip(*[input[i:] for i in range(n)])

occurrence_dict = {k: 0 for k in keywords}

for idx,row in tqdm(congress_df.iterrows(), total=len(congress_df.index)):
    doc = row['text'].lower()
    # get ngrams with keyword in them
    ITER = False
    for keyword in keywords:
        if keyword in doc:
            ITER = True
            occurrence_dict[keyword] += 1
    
    # only want to tokenize/process if we have keyword
    if ITER:
        words = word_tokenize(doc)
        ngrams = get_ngrams(words, n=N)
        for ngram in ngrams:
            for keyword in keywords:
                if keyword in ngram:
                    ngram_str = '_'.join(ngram)
                    if ngram_str not in ngram_dict[keyword].keys():
                        ngram_dict[keyword][ngram_str] = 1
                    else:
                        ngram_dict[keyword][ngram_str] += 1

print("Occurrence dict:")
print(occurrence_dict)

print("Got ngrams, now finding top phrases for each keyword")
for k in keywords:
    sorted_ngram_list = sorted(ngram_dict[k], key=ngram_dict[k].get, reverse=True)
    print(f"For token {k}, top 20 ngrams are:")
    for ngram in sorted_ngram_list[:20]:
        print(f"{ngram}: count of {ngram_dict[k][ngram]}")

with open(f'/data/laviniad/congress_errata/ngram_dict_{N}.json', 'w') as f:
    json.dump(ngram_dict, f)
