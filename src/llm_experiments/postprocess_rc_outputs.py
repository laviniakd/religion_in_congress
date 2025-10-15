import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import data.bible_utils as bible_utils
import pickle as pkl
from tqdm import tqdm

tqdm.pandas()

from postprocess_llm_outputs import process_rc_response, process_rc_kw_response, load_data_together

# load data

PATH = "/data/laviniad/congress_errata/llm_outputs/old_rc_416/"
LOG_PATH = "/home/laviniad/projects/religion_in_congress/data/llm_logs/old_logs/"
OUT_PATH = "/data/laviniad/congress_errata/llm_outputs/religious_classification.pkl"

print("Loading data...")
results_data = load_data_together(PATH, LOG_PATH)
congress_df = pd.read_json("/data/laviniad/congress_errata/congress_df.json")
congress_df["idx"] = congress_df.index.copy()

debug_sample_path = "/data/laviniad/congress_errata/debug_sample.json"
sample_df = pd.read_json(debug_sample_path)
sample_df["idx"] = sample_df.index.copy()

print("Data loaded.")

# add congress_df features to sample_df based on "idx"
sample_df = sample_df.merge(congress_df, on=["text", "bio_id", "date", "year", "congress_num", "speaker"], how="left", suffixes=("", "_congress"))


print("Processing data...")
for run_id, data in tqdm(results_data.items(), total=len(results_data)):
    df = data["df"]
    # label
    prompt_path = data["log"]["prompt_path"]

    if prompt_path == "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/RELIGIOUS_THEME_PROMPT.txt":
        df["llm_label"] = df["llm_response"].apply(lambda x: process_rc_response(x))
    elif prompt_path == "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/RELIGIOUS_THEME_KEYWORDS_PROMPT.txt":
        responses = df["llm_response"].apply(lambda x: process_rc_kw_response(x))
        df["llm_label"] = [response[0] for response in responses]
        df["llm_keywords"] = [response[1] for response in responses]
    
    df["idx"] = df.index.copy()
    #df = df[df["idx"].isin(congress_df["idx"])]
    # add cols from sample_df that are not in df
    df = df.merge(sample_df, suffixes=("", "_sample"))
    results_data[run_id]["df"] = df

print(f"Processed. This corresponds to {len(results_data)} runs.")

# save to file
print(f"Saving data to {OUT_PATH}")
print("...")
with open(OUT_PATH, "wb") as f:
    pkl.dump(results_data, f)