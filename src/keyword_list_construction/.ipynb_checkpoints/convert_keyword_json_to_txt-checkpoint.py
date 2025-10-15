import json

NUM = 150
INPUT_PATH = '/data/laviniad/sermons-ir/log_odds/final/congress_sermon_odds.json'
OUTPUT_PATH = '/home/laviniad/projects/religion_in_congress/src/multi-feature-use/keywords_from_congress.txt'
with open(INPUT_PATH) as f:
    data = json.load(f)
    
word_list = list(data)[:NUM]

with open(OUTPUT_PATH, 'w') as f:
    f.writelines([w + '\n' for w in word_list])
