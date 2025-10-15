# imports
import pandas as pd

from data.congress_utils import load_full_df_from_raw
import numpy as np
from pprint import pprint
from collections import Counter
import json
from tqdm import tqdm
import time
import tiktoken
import openai
import anthropic

from src.llm_experiments.llm_utils import label_text, get_clients

tqdm.pandas()


#nlp = spacy.load("en_core_web_sm")
#config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
#nlp.add_pipe("sentencizer", config=config)
#from spacy.lang.en import stop_words
import argparse

#stop_words = stop_words.STOP_WORDS

DEBUG = False
add_topic = True
USE_ONLY_GOD_TALK = False
ADD_KEYWORD_BOOLS = True
REMOVE_TEXT = False
MAX_LENGTH = None
MAX_NEW_TOKENS = 50
DEBUG_SIZE = 1000

# constants
# Argument parser
parser = argparse.ArgumentParser(description="Process congressional data.")
parser.add_argument('--congress_path', type=str, default="/data/corpora/congressional-record/", help='Path to the congressional data.')
parser.add_argument('--output_prefix', type=str, default="/data/laviniad/congress_errata/llm_outputs/run_", help='Prefix for the output file.')
parser.add_argument('--prompt_path', type=str, default="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/RELIGIOUS_THEME_PROMPT.txt", help='Path to prompt string.')
parser.add_argument('--model', type=str, default="gpt-4o", help='Model to use (e.g., gpt-4o, claude-3-5-sonnet-20240620).')
parser.add_argument('--logging_path', type=str, default="/home/laviniad/projects/religion_in_congress/data/llm_logs", help='Path to logs.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with a sample of the data.')
parser.add_argument('--debug_data_file', type=str, default="/data/laviniad/congress_errata/debug_sample.json", help='Path to debug data file if debug mode is enabled.')

# add temperature, sample, top_p
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation.')
parser.add_argument('--sample', action='store_true', help='Use sampling for text generation.')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for text generation.')

args = parser.parse_args()

log_dict = {}

# Constants
congress_path = args.congress_path
output_prefix = args.output_prefix

DEBUG = args.debug
SAMPLE = args.sample if '--sample' in args else False
TEMPERATURE = args.temperature
TOP_P = args.top_p
MODEL = args.model

log_dict['timestamp'] = time.time()

log_dict['congress_path'] = congress_path
log_dict['output_prefix'] = output_prefix
log_dict['prompt_path'] = args.prompt_path
log_dict['model'] = MODEL
log_dict['logging_path'] = args.logging_path
log_dict['debug'] = DEBUG
log_dict['sample'] = SAMPLE
log_dict['temperature'] = TEMPERATURE
log_dict['top_p'] = TOP_P


# data loading
print("***** Loading congressional data and speaker covariates *****")
congress_df = None
if not DEBUG:
    congress_df = load_full_df_from_raw(congress_path, remove_procedural_speeches=True)
else:
    if args.debug_data_file:
        congress_df = pd.read_json(args.debug_data_file)
    else:
        congress_df = congress_df.sample(DEBUG_SIZE)
        congress_df.to_json("/data/laviniad/congress_errata/debug_sample.json")
        log_dict['debug_size'] = DEBUG_SIZE

print("Columns in congress_df: ", congress_df.columns)

if not DEBUG:
    assert(2017 in congress_df['year'].unique()) # check that data loading is working like it should

def calculate_tiktoken_lengths(congress_df):
    print("***** Collecting tiktoken lengths *****")

    for idx, row in tqdm(congress_df.iterrows(), total=len(congress_df.index)):
        if row['text'] is not None:
            congress_df.at[idx, 'tiktoken_length'] = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(row['text']))
        else:
            congress_df.at[idx, 'tiktoken_length'] = 0
    
    congress_df['tiktoken_length'] = congress_df['tiktoken_length'].astype(int)
    congress_df['tiktoken_length'].to_json("/data/laviniad/congress_errata/tiktoken_lengths.json")

#calculate_tiktoken_lengths(congress_df)

print("***** LLM labeling *****")

# load prompt
with open(args.prompt_path, 'r') as f:
    prompt = f.read()

# Set up client for OpenAI or Anthropic based on model name
openai_client, anthropic_client = get_clients(MODEL)
log_dict['max_new_tokens'] = MAX_NEW_TOKENS

def label_text_here(text):
    return label_text(prompt, text, openai_client if openai_client else anthropic_client, MODEL, temp=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS, sample=False)

# apply labeling function to text column
print("Beginning labeling at time ", time.time())

congress_df['llm_response'] = congress_df['text'].progress_apply(label_text_here)
print("Labeling complete at time ", time.time())

# timestamp in utc
timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(log_dict['timestamp']))
log_dict['timestamp'] = timestamp

# Keep only the text and 'llm_response' columns
congress_df = congress_df[['text', 'llm_response']]

congress_path = output_prefix + f"outputs_{timestamp}.json"
congress_df.to_json(congress_path)
print(f"Dumped to {congress_path}")

logs_path = args.logging_path + f"/llm_log_{timestamp}.json"
with open(logs_path, 'w') as f:
    json.dump(log_dict, f)
print(f"Dumped logs to {logs_path}")

print("***** Dumping logs *****")
print("***** Done *****")