import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import nltk
import json
import matplotlib.ticker as ticker

palette = {'Democrat': 'blue', 'Republican': 'red', 'New Progressive': 'grey', 'Popular Democrat': 'grey', 'Independence Party (Minnesota)': 'grey', 'Anti-Jacksonian': 'grey', 'Independent': 'green', 'unknown': 'black', 'Democrat Farmer Labor': 'blue'}
sns.set(context="notebook", font_scale=3, rc={'figure.figsize':(16,12), 'font.weight': 'normal'}, style='whitegrid')

with open('/home/laviniad/projects/religion_in_congress/data/wordcounts/wordcounts.json') as f:
    result = json.load(f)
    print("Got results from /home/laviniad/projects/religion_in_congress/data/wordcounts/wordcounts.json")

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
print("Saved file to /home/laviniad/projects/religion_in_congress/notebooks/plots/new_plots/word_counts.pdf")
