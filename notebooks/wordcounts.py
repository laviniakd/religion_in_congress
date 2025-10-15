# keyword counts!

import sys
print(sys.path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import nltk

tqdm.pandas()

DATA_PATH = "/data/laviniad/congress_errata/congress_df.json"
df = pd.read_json(DATA_PATH)
df['date'] = pd.to_datetime(df['date'].apply(str), format='%Y%m%d')
df['month'] = df['date'].dt.month

def is_dem_rep(t):
    return (t == 'Democrat') or (t == 'Republican')

def not_procedural_unknown_topic(x):
    return (not ('Procedural' in x)) and (not ('Unknown' in x))

def is_long(x, thresh=5): # takes string, whitespace heuristic
    return len(x.split()) > thresh

def is_in_range(x):
    return x in range(1995, 2023)

df = df[df['party'].progress_apply(is_dem_rep)]
#df = df[df['main_topic'].progress_apply(not_procedural_unknown_topic)]
df = df[df['year'].progress_apply(is_in_range)]
df = df[df['text'].progress_apply(is_long)]

palette = {'Democrat': 'blue', 'Republican': 'red', 'New Progressive': 'grey', 'Popular Democrat': 'grey', 'Independence Party (Minnesota)': 'grey', 'Anti-Jacksonian': 'grey', 'Independent': 'green', 'unknown': 'black', 'Democrat Farmer Labor': 'blue'}
sns.set(context="notebook", font_scale=3, rc={'figure.figsize':(16,12), 'font.weight': 'normal'}, style='whitegrid')

# Get the current project directory
project_path = os.getcwd()  # or set explicitly, e.g., "/path/to/your/project"
sys.path = ['/home/laviniad/projects/religion_in_congress']

from data.data_utils import get_lexical_overlap
print(f"File of data is {data.__file__}")
import collections

# get counts
objs = df['text'].progress_apply(lambda x: get_lexical_overlap(x)) # each obj is count / len(speech), count_dict, type_dict, toked_length

# aggregate count_dicts
count_dict = collections.Counter()
for d in objs:
    count_dict.update(d[1])
     
result = dict(count_dict)

print("Counter..."
      f"Total words: {sum(result.values())}, "
      f"Unique words: {len(result)}")

# save result to /home/laviniad/projects/religion_in_congress/data/wordcounts
import json
with open('/home/laviniad/projects/religion_in_congress/data/wordcounts/wordcounts.json', 'w') as f:
    json.dump(result, f)
    print("Saved results to /home/laviniad/projects/religion_in_congress/data/wordcounts/wordcounts.json")

# plot results
import matplotlib.ticker as ticker
agg_pray = False

if agg_pray:
    PRAY_WORDS = ['pray', 'praying', 'prayer', 'prayed']
    results_with_pray_forms_combined = result.copy()
    results_with_pray_forms_combined['pray*'] = sum([results_with_pray_forms_combined.get(w, 0) for w in PRAY_WORDS])
    result = dict(sorted(results_with_pray_forms_combined.items(), key=lambda item: item[1], reverse=True)[:25])
else:
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True)[:25])

words, counts = list(result.keys()), list(result.values())
print("Words:")
print(words)
print("Counts:")
print(counts)
counts = [np.log10(c) for c in counts]
print("Counts logged:")
print(counts)

sns.barplot(y=words,x=counts,hue=words,palette='viridis')
plt.xlabel('Count')

#plt.gca().set_xticks(np.log10([1, 10, 100, 1000, 10000, 100000]))  # Set the ticks to match the original values
plt.gca().set_xticklabels(['1', '10', '100', '1,000', '10,000', '100,000'])  # Change the ticks' names
plt.savefig("/home/laviniad/projects/religion_in_congress/notebooks/plots/new_plots/word_counts.pdf", format='pdf', dpi=300, bbox_inches='tight')