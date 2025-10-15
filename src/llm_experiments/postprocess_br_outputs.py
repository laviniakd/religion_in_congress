import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import data.bible_utils as bible_utils
import pickle as pkl

from postprocess_llm_outputs import *

TEST = True
EXPECTED_RUNS = 1

LOG_DIR = "/home/laviniad/projects/religion_in_congress/data/llm_logs/"
LABEL_DIR = "/data/laviniad/congress_errata/llm_outputs/biblical_reference/"
PROMPT_VERSION_DIR = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/best_br/"
PROMPT_METADATA_DIR = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_metadata/best_br/"
OUTPUT_PATH = "/data/laviniad/congress_errata/llm_outputs/biblical_reference.pkl"

MIN_DATE = None
if TEST:
    MIN_DATE = "20250503"
    OUTPUT_PATH = OUTPUT_PATH.replace(".pkl", "_final.pkl")

out_defaults = ["YES", "NO"]

logs = load_logs(LOG_DIR)
labels = load_br_labels([LABEL_DIR])
prompt_metadata = load_prompt_metadata(PROMPT_METADATA_DIR)

logs = {k: logs[k] for k in logs.keys() if k in labels.keys()}
labels = {k: labels[k] for k in labels.keys() if k in logs.keys()}
print(f"Loaded {len(logs)} logs and {len(labels)} labels")
# add prompt metadata to logs
for k in logs.keys():
    prompt_path_bare = logs[k]['prompt_path'].split('/')[-1].split('.txt')[0]
    logs[k]['prompt_metadata'] = prompt_metadata[prompt_path_bare] if prompt_path_bare in prompt_metadata.keys() else None


#print("Number of logs before filtering:", len(logs))
#logs = {k: logs[k] for k in logs.keys() if 'PROMPT' not in logs[k]['prompt_path']}
#print("Number of logs after filtering:", len(logs))
# filter labels to only include those that are in logs
#print("Number of label files after filtering:", len(labels))
#labels = {k: labels[k] for k in labels.keys() if k in logs.keys()}
#print("Number of label files after filtering:", len(labels))


def create_unified_datastructure(logs, labels, out_defaults):
    processed_labels = {}
    for timestamp, label_data in labels.items():
        df = label_data
        if not logs[timestamp]['prompt_metadata']:
            #print("No metadata found for timestamp:", timestamp)
            #print("Relevant file is in:", logs[timestamp]['prompt_path'])
            continue

        out = logs[timestamp]['prompt_metadata']['output_format'] if 'output_format' in logs[timestamp]['prompt_metadata'] else None
        date_timestamp = logs[timestamp]['timestamp'].split("_")[0]
        if MIN_DATE is not None and int(date_timestamp) < int(MIN_DATE):
            continue

        if (timestamp not in logs) or ('prompt_path' not in logs[timestamp]):
            #print("Timestamp not found:", timestamp)
            continue

        if logs[timestamp]['prompt_path'] == "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/PROMPT_DETECT.txt":
        #print(df['llm_response'].iloc[34])
            df['label'] = df['llm_response'].apply(lambda x: process_br_detect_response(x, output_format=out))
        # convert None to np.nan
            df['label'] = df['label'].replace({None: np.nan})
        else:
        #print(df['llm_response'].iloc[34])
            is_json = logs[timestamp]['prompt_metadata']['json_format']
            label_and_verse_list = df['llm_response'].apply(lambda x: process_br_detect_id_response(x, output_format=out, json_formatted=is_json))
            df['label'] = [(item[0] if item is not None else None) for item in label_and_verse_list]
            print("DF label sample:", df['label'].sample(10))
            df['label'] = df['label'].replace({None: np.nan})
            df['verse'] = [(item[1] if item is not None else None) for item in label_and_verse_list]

        processed_labels[timestamp] = {}
        processed_labels[timestamp]['df'] = df
        processed_labels[timestamp]['log'] = logs[timestamp]

        # if many of the labels are None, print the text of the output ('llm_response') of the first 10
        if df['label'].isna().sum() > 0.5 * len(df):
            print("Many labels are None for timestamp:", timestamp)
            #print("Output text:", df['llm_response'].iloc[0:10])
            #def process_for_out(t):
            #    t = t.strip().split("\n")
            #    t = [x.strip() for x in t if x.strip() != ""]
            #print("Processed output text:", df['llm_response'].apply(process_for_out)[0:10])

    processed_labels = {k: processed_labels[k] for k in processed_labels.keys() if k in logs.keys()}
    return processed_labels


#reference_df = load_reference_df()
processed_labels = create_unified_datastructure(logs, labels, out_defaults)

model_counts = {}
for k, v in processed_labels.items():
    model_name = v['log']['model'] if 'model' in v['log'] else v['log']['hf_model']
    if model_name not in model_counts:
        model_counts[model_name] = 0
    model_counts[model_name] += 1
print("Number of runs for each model:")
for k, v in model_counts.items():
    print(f"{k}: {v}")

for m in model_counts.keys():
    if model_counts[m] != EXPECTED_RUNS:
        print(f"Model {m} dooes not have 90 runs, but {model_counts[m]} runs")

        # remove the model from processed_labels
        processed_labels = {k: processed_labels[k] for k in processed_labels.keys() if processed_labels[k]['log']['model'] != m}

with open(OUTPUT_PATH, "wb") as f:
    pkl.dump(processed_labels, f)
