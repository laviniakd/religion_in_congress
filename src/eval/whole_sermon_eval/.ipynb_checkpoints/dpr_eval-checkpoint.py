import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch

# EXTREMELY based on Katherine Thai's code for RELiC

from data.bible_utils import bible_helper, build_candidates
from data.data_utils import get_verse_dicts
from src.eval.eval_utils import load_paired_data, load_whole_sermon_data
from src.relic_utils.utils import build_lit_instance, print_results, NUM_SENTS

from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

SERMON_DF_PATH = '/data/laviniad/sermons-ir/sermoncentral/fsm/'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--num_examples', default=100, type=int)
parser.add_argument('--split', default="train", type=str)
parser.add_argument('--output_dir', default="/home/laviniad/projects/religion_in_congress/results/whole_sermon/dpr/", type=str)
parser.add_argument('--eval_small', action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

context_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
query_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
query_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_model.cuda()
query_model.cuda()

BATCH_SIZE = 100

bible_df = bible_helper(BIBLE_PATH)
verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
verse_text_dict = {k: v.lower() for k,v in verse_text_dict.items()}

text_verse_dict = {v: k for k,v in verse_text_dict.items()}

cands = build_candidates(bible_df)
verse_to_idx = {v.lower(): k for k,v in enumerate(cands)}

print(f"{'-' * 25} Now evaluating: {'-' * 25}")
sermon_df, sermon_verse_data, sermon_texts = load_whole_sermon_data(SERMON_DF_PATH, args.split, verse_text_dict, verse_to_idx, args.num_examples, fuzzy_citation_dict)
full_len = float(len(commentary_verse_df.index))

results = {
        "mean_rank": [],
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "recall@50": [],
        "recall@100": [],
        "num_candidates": []
        }

total = 0
submission_data = {}

prefixes = []
for inst_num in range(0, len(sermon_texts), BATCH_SIZE):
    with torch.inference_mode():
        dpr_tensors = query_tokenizer([i[0] for i in sermon_texts[inst_num:inst_num + BATCH_SIZE]], return_tensors="pt", padding=True, truncation=True, max_length=512)
        for k, v in dpr_tensors.items():
            dpr_tensors[k] = v.cuda()
        prefixes.append(query_model(**dpr_tensors).pooler_output)
prefixes = torch.cat(prefixes, dim=0)

suffixes = []
for inst_num in range(0, len(cands), BATCH_SIZE):
    with torch.inference_mode():
        dpr_tensors = context_tokenizer(cands[inst_num:inst_num + BATCH_SIZE], return_tensors="pt", padding=True, truncation=True, max_length=512)
        for k, v in dpr_tensors.items():
            dpr_tensors[k] = v.cuda()
        suffixes.append(context_model(**dpr_tensors).pooler_output)
suffixes = torch.cat(suffixes, dim=0)

with torch.inference_mode():
    similarities = torch.matmul(prefixes, suffixes.t())
    sorted_scores = torch.sort(similarities, dim=1, descending=True)
    sorted_score_idx, sorted_score_vals = sorted_scores.indices, sorted_scores.values

ranks = []
for qnum, (quote_data, context) in tqdm(enumerate(zip(sermon_verse_data, sermon_texts)), total=len(sermon_texts)):
    quote_id, verse = quote_data[0], quote_data[1]
    gold_rank = int((sorted_score_idx[qnum] == quote_id).nonzero().squeeze().cpu().numpy())
    ranks.append(gold_rank)

results["mean_rank"].extend(ranks)
results["recall@1"].extend([x <= 1 for x in ranks])
results["recall@3"].extend([x <= 3 for x in ranks])
results["recall@5"].extend([x <= 5 for x in ranks])
results["recall@10"].extend([x <= 10 for x in ranks])
results["recall@50"].extend([x <= 50 for x in ranks])
results["recall@100"].extend([x <= 100 for x in ranks])

print_results(results)
with open(f"{args.output_dir}/{args.num_examples}instances.json", 'w') as f:
    json.dump(results, f)
