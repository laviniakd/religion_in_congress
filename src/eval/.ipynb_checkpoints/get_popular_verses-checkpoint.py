import pandas as pd
from collections import Counter
from ast import literal_eval
from tqdm import tqdm
import json

## defining popularity wrt training set only
## ids are the indices in /fsm/train.csv
IN_PATH = '/data/laviniad/sermons-ir/whole-sermon-dfs/fsm/train.csv'
OUT_PATH = '/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv'
VERSE_TO_SERMON_ID = '/data/laviniad/sermons-ir/sermoncentral/matched_verses_to_sermon_ids.json'
TOP_NUM = 250

OUTPUT_COUNT = False

# load data
df = pd.read_csv(IN_PATH)
df['verses'] = df['verses'].apply(literal_eval)

verse_lists = list(df['verses'])
flattened = [item for sublist in verse_lists for item in sublist]

if OUTPUT_COUNT:
    print("Counting verses...")
    verse_count_dict = dict(Counter(flattened))
    vc_df = pd.DataFrame(list(verse_count_dict.items()), columns=['verse', 'count'])
    vc_df['proportion'] = vc_df['count'] / float(len(flattened))
    vc_df.sort_values('count', ascending=False, inplace=True)
    vc_df.to_csv(OUT_PATH)
    print(f"Dumped verse counts to {OUT_PATH}")

print("Now getting verse-sermon id linking dict for future convenience...")
verse_id_dict = {v: [] for v in set(flattened)}
for r in tqdm(df.itertuples()):
    #print(r)
    verse_list = r.verses
    for v in verse_list:
        verse_id_dict[v].append(r.index)


with open(VERSE_TO_SERMON_ID, 'w') as f:
    json.dump(verse_id_dict, f)

