## filters dfs to ensure all english and separates into train, dev, and test
import json
from pprint import pprint
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import pycld2

FULL_SERMON_PATH = '/data/laviniad/sermons-ir/sermoncentral/references/raw_references/refdf_fsm90-250.json'
OUT_PATH = '/data/laviniad/sermons-ir/sermoncentral/references/'
train_perc = 0.8
dev_perc = 0.1
test_perc = 0.1


def is_english(text):
    try:
        return pycld2.detect(text)[2][0][1] == 'en'
    except pycld2.error:
        return False


with open(FULL_SERMON_PATH) as f:
    full_data = json.load(f)

full_df = []
for sermon in tqdm(full_data):
    for ref in sermon[1]:
        full_df.append({'verse': ref['verse'],
                        'verse_text': ref['verse_text'],
                        'sermon_idx': sermon[0], 
                        'text': ref['original_text']})

full_df = pd.DataFrame(full_df)
full_df_filtered = full_df[full_df['text'].apply(is_english)]

english_portion = len(full_df_filtered.index) / len(full_df.index)
print("English portion of full df: " + str(english_portion))

train, test = train_test_split(full_df, test_size=test_perc)

train, dev = train_test_split(train, test_size=float(1 / 9))

print("Number of instances in train: ", len(train.index))
print("Number of instances in dev: ", len(dev.index))
print("Number of instances in test: ", len(test.index))

train.to_csv(OUT_PATH + 'train.csv')
dev.to_csv(OUT_PATH + 'dev.csv')
test.to_csv(OUT_PATH + 'test.csv')
