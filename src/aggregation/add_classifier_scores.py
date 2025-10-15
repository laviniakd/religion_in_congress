import pandas as pd
from tqdm import tqdm
import numpy as np
import json

DEBUG = False
LR_ONLY = True
REMOVE_TEXT = False

PREEXISTING_DATA_PATH = "/data/laviniad/congress_errata/congress_df.json"
OUTPUT_DATA_PATH = "/data/laviniad/congress_errata/classed_congress_df.json"
PATH = "/data/laviniad/congress_errata/idx_to_classifier_output.json"
LOGISTIC_PATH = "/data/laviniad/congress_errata/idx_to_lr_classifier_output.json"

df = pd.read_json(PREEXISTING_DATA_PATH)
print(f"{len(df.index)} indices in the dataframe")
with open(PATH, 'r') as f:
    idx_to_classifier_output = json.load(f)

if DEBUG:
    df = df.sample(10000) # should have docs that were labeled in this sample

if not LR_ONLY:
    print("Loading RoBERTa classifier output")
    in_index = 0
    # add the classifier output to the dataframe
    for idx, output in tqdm(idx_to_classifier_output.items()):
        idx = int(idx)
        if idx in df.index:
            in_index += 1

            if isinstance(output, list): # list of dicts
                prob_religious_arr = [x['probs'][1] for x in output] # 1 is sermon
                prop_relig = np.mean([int(x['label'] == 'religious') for x in output])
                df.at[idx, 'classifier_label_prop_religious'] = prop_relig # threshold at 0.5
                df.at[idx, 'max_classifier_prob'] = np.max(prob_religious_arr)
                df.at[idx, 'avg_classifier_prob'] = np.mean(prob_religious_arr)
            else:
                df.at[idx, 'classifier_label_prop_religious'] = None
                df.at[idx, 'max_classifier_prob'] = None
                df.at[idx, 'avg_classifier_prob'] = None
        else:
            print(f"Index {idx} not in dataframe")

    print(f"Added classifier output to {in_index} indices, or {in_index/len(df.index)} of the dataframe.")

print("Loading logistic regression classifier output")
with open(LOGISTIC_PATH, 'r') as f:
    idx_to_lr_output = json.load(f)

in_index = 0
# add the classifier output to the dataframe
for idx, output in tqdm(idx_to_lr_output.items()):
    idx = int(idx)
    if idx in df.index:
        in_index += 1

        #print("output: ", str(output))

        if isinstance(output, list): # list of dicts
            prob_religious_arr = [x['probs'][0][1] for x in output] # 1 is sermon
            prop_relig = np.mean([int(x['label'] == 'religious') for x in output])
            df.at[idx, 'lr_label_prop_religious'] = prop_relig # threshold at 0.5
            df.at[idx, 'max_lr_prob'] = np.max(prob_religious_arr)
            df.at[idx, 'avg_lr_prob'] = np.mean(prob_religious_arr)
        else:
            df.at[idx, 'lr_label_prop_religious'] = None
            df.at[idx, 'max_lr_prob'] = None
            df.at[idx, 'avg_lr_prob'] = None
    else:
        print(f"Index {idx} not in dataframe")

print(f"Added classifier output to {in_index} indices, or {in_index/len(df.index)} of the dataframe.")

if REMOVE_TEXT:
    print("Removing text")
    df = df.drop(columns=['text'])

# save the dataframe
df.to_json(OUTPUT_DATA_PATH)
print("Saved the dataframe with the classifier output")
