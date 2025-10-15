# imports
import pandas as pd

#from data.congress_utils import load_full_df_from_raw
import numpy as np
from pprint import pprint
from collections import Counter
import json
from tqdm import tqdm
import time
import tiktoken
import os
import re

hf_token = os.getenv("HF_TOKEN")
print("HF_TOKEN: ", hf_token)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from accelerate import infer_auto_device_map

tqdm.pandas()

import spacy
from spacy.pipeline import Sentencizer

nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
nlp.add_pipe("sentencizer", config=config)
from spacy.lang.en import stop_words
import argparse
stop_words = stop_words.STOP_WORDS

add_topic = True
USE_ONLY_GOD_TALK = False
ADD_KEYWORD_BOOLS = True
REMOVE_TEXT = False
MAX_LENGTH = None
MAX_NEW_TOKENS = 50
DEBUG_SIZE = 1000

os.environ["TRANSFORMERS_VERBOSITY"] = "error" # necessary to suppress warnings about ids
max_memory = {
    0: "48GiB",  # A6000
    1: "48GiB",  # A6000
    2: "22GiB",  # A5000
} # device settings
# set visible devices to be 1 and 2, not 0
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
#print("Visible devices: ", os.getenv("CUDA_VISIBLE_DEVICES"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# constants
# Argument parser
parser = argparse.ArgumentParser(description="Process congressional data.")
parser.add_argument('--congress_path', type=str, default="/data/corpora/congressional-record/", help='Path to the congressional data.')
parser.add_argument('--output_prefix', type=str, default="/data/laviniad/congress_errata/llm_outputs/biblical_reference/run_", help='Prefix for the output file.')
parser.add_argument('--prompt_path', type=str, default="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/PROMPT_DETECT.txt", help='Path to prompt string.')
parser.add_argument('--hf_model', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='LLM model to use.')
parser.add_argument('--logging_path', type=str, default="/home/laviniad/projects/religion_in_congress/data/llm_logs", help='Path to logs.')
parser.add_argument('--device', type=str, default="cuda:1", help='GPU ID for model.')
parser.add_argument('--data_path', default='/home/laviniad/projects/religion_in_congress/data/dallas_lavinia_combined_annotations.csv', help='Validation data path.')

# add temperature, sample, top_p
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation.')
parser.add_argument('--sample', action='store_true', help='Use sampling for text generation.')
parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for text generation.')

args = parser.parse_args()

log_dict = {}

# Constants
congress_path = args.congress_path
output_prefix = args.output_prefix

SAMPLE = args.sample
TEMPERATURE = args.temperature
TOP_P = args.top_p

log_dict['timestamp'] = time.time()

log_dict['congress_path'] = congress_path
log_dict['output_prefix'] = output_prefix
log_dict['prompt_path'] = args.prompt_path
log_dict['hf_model'] = args.hf_model
log_dict['logging_path'] = args.logging_path
log_dict['device'] = args.device
log_dict['sample'] = SAMPLE
log_dict['temperature'] = TEMPERATURE
log_dict['top_p'] = TOP_P

num_model_params_in_billions = re.search(r'\-(\d+)B\-', args.hf_model).group(0)
num_model_params_in_billions = int(num_model_params_in_billions.replace("-", "").replace("B", ""))
log_dict['num_model_params'] = num_model_params_in_billions 


# data loading
print("***** Loading reference data and speaker covariates *****")
reference_df = pd.read_csv(args.data_path)
print("Number of instances in reference_df: ", len(reference_df))

print("Columns in reference_df: ", reference_df.columns)


def calculate_tiktoken_lengths(reference_df):
    print("***** Collecting tiktoken lengths *****")

    for idx, row in tqdm(reference_df.iterrows(), total=len(reference_df.index)):
        if row['text'] is not None:
            reference_df.at[idx, 'tiktoken_length'] = len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(row['text']))
        else:
            reference_df.at[idx, 'tiktoken_length'] = 0
    
    reference_df['tiktoken_length'] = reference_df['tiktoken_length'].astype(int)
    reference_df['tiktoken_length'].to_json("/data/laviniad/congress_errata/br_tiktoken_lengths.json")

#calculate_tiktoken_lengths(reference_df)

print("***** LLM labeling *****")

# load model: 

def load_config():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        #bnb_4bit_use_double_quant=True,
        #bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    return quant_config

def load_device_map(model):
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer"],  # Don't split individual transformer blocks
        dtype=torch.float16
    )
    return device_map


print("Available devices: ", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))
print(torch.cuda.get_device_name(2))
    
if 'cpu' in args.device:
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, token=hf_token, low_cpu_mem_usage=True, torch_dtype=torch.float16).eval()
    print("Model loaded on CPU")

elif num_model_params_in_billions > 30:
    torch.cuda.empty_cache()
    quant_config = load_config()
    #device_map = load_device_map(model)

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        offload_folder="/data/laviniad/congress_errata/offload",
    )


    #model = AutoModelForCausalLM.from_pretrained(
    #    args.hf_model,
    #    quantization_config=quant_config,
    #    device_map=device_map,
    #    trust_remote_code=True,
    #    low_cpu_mem_usage=True
    #)
else:
    if 'cuda' in args.device:
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, token=hf_token, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(args.device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, token=hf_token, low_cpu_mem_usage=True, torch_dtype=torch.float16).eval()
#print(model.device)

tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
model.generation_config.pad_token_id = tokenizer.pad_token_id


# load prompt
with open(args.prompt_path, 'r') as f:
    prompt = f.read()

# load pipeline

if num_model_params_in_billions > 30:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        max_length=MAX_LENGTH,
        #top_k=50,
    )
elif SAMPLE and args.device == "auto":
    # cuda visible devices are 0 and 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    print("Visible devices: ", os.getenv("CUDA_VISIBLE_DEVICES"))
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        do_sample=True,
        torch_dtype=torch.float16,
        device_map="auto",  # Use GPU if available
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P
    )
elif SAMPLE:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=TEMPERATURE,
        do_sample=True,
        torch_dtype=torch.float16,
        device=args.device,  # Use GPU if available
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P
    )
else:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        #max_length=MAX_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        #top_p=0.9,
        #top_k=50,
    )

log_dict['max_length'] = MAX_LENGTH
log_dict['max_new_tokens'] = MAX_NEW_TOKENS

# def labeling

# prompt_code = {"RELIGIOUS": 1, "SECULAR": 0}

def label_text(text):
    response = pipe(prompt + text, num_return_sequences=1)
    # Extract the generated text from the response
    generated_text = response[0]['generated_text']
    # Extract the text after the prompt + text
    generated_text = generated_text[len(prompt + text):].strip()
    return generated_text

# apply labeling function to text column
print("Beginning labeling at time ", time.time())

reference_df['llm_response'] = reference_df['text'].progress_apply(lambda x: label_text(x) if x is not None else None)
print("Labeling complete at time ", time.time())

# timestamp in utc
timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(log_dict['timestamp']))
log_dict['timestamp'] = timestamp

# Keep only the text and 'llm_response' columns
reference_df = reference_df[['text', 'llm_response']]

log_dict['prompt'] = prompt

congress_path = output_prefix + f"outputs_{timestamp}.json"
reference_df.to_json(congress_path)
print(f"Dumped to {congress_path}")

logs_path = args.logging_path + f"/llm_log_{timestamp}.json"
with open(logs_path, 'w') as f:
    json.dump(log_dict, f)
print(f"Dumped logs to {logs_path}")

print("***** Dumping logs *****")
print("***** Done *****")
