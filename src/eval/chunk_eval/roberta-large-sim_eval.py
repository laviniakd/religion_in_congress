import os
import torch
import argparse
from tqdm import tqdm
import json
from transformers import RobertaModel, RobertaTokenizer
from data.bible_utils import bible_helper, build_candidates
from data.data_utils import get_verse_dicts
from src.eval.eval_utils import load_paired_data

# EXTREMELY based on Katherine Thai's code for RELiC

from src.relic_utils.utils import print_results

BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
CV_DF_PATH = '/shared/3/projects/sermons-ir/commentary-verse-dfs/'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="RELiC", type=str)
parser.add_argument('--split', default="train", type=str)
parser.add_argument('--num_examples', default=100, type=int)
parser.add_argument('--left_sents', default=4, type=int)
parser.add_argument('--right_sents', default=4, type=int)
parser.add_argument('--output_dir', default="/home/laviniad/projects/religion_in_congress/results/cos_sim/", type=str)
parser.add_argument('--eval_small', action='store_true')
parser.add_argument('--rewrite_cache', action='store_true')
parser.add_argument('--cache_scores', action='store_true')
parser.add_argument('--cache', action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

BATCH_SIZE = 100
total = 0
submission_data = {}

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')
model.cuda()


COMMENTARY_VERSE = '/shared/3/projects/sermons-ir/commentary-verse-dfs/'
FUZZY_CITATION_DICT_PATH = '/shared/3/projects/sermons-ir/fuzzy_citation_dict.json'

bible_df = bible_helper(BIBLE_PATH)
verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
verse_text_dict = {k: v.lower() for k,v in verse_text_dict.items()}

text_verse_dict = {v: k for k,v in verse_text_dict.items()}

context_windows = [(1,1), (2,2), (3,3), (4,4),
                   (0,1), (1,0), (2,0), (0,2),
                   (0,3), (3,0), (4,0), (0,4)]

cands = build_candidates(bible_df)
verse_to_idx = {v.lower(): k for k,v in enumerate(cands)}
cos = torch.nn.CosineSimilarity(dim=-1)

for cw in context_windows:

    print(f"{'-' * 25} Now evaluating with context window: {cw} {'-' * 25}")
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

    commentary_verse_df, all_masked_quotes, all_contexts = load_paired_data(COMMENTARY_VERSE, args.split,
                                                                            cw, verse_text_dict, verse_to_idx,
                                                                            args.num_examples, input_is_text_not_cite=True)

    full_len = float(len(commentary_verse_df.index))

    prefixes = []
    for inst_num in range(0, len(all_contexts), BATCH_SIZE):
        with torch.inference_mode():
            roberta_tensors = tokenizer([i[0] for i in all_contexts[inst_num:inst_num + BATCH_SIZE]], return_tensors="pt",
                                    padding=True, truncation=True, max_length=512)
            for k, v in roberta_tensors.items():
                roberta_tensors[k] = v.cuda()
            prefixes.append(model(**roberta_tensors).pooler_output)

    prefixes = torch.cat(prefixes, dim=0)

    #print(f"cands: {cands}")
    suffixes = []
    for inst_num in range(0, len(cands), BATCH_SIZE):
        with torch.inference_mode():
            roberta_tensors = tokenizer(cands[inst_num:inst_num + BATCH_SIZE], return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)

            #print(f"Tensor shape: {roberta_tensors.shape}")
            for k, v in roberta_tensors.items():
                roberta_tensors[k] = v.cuda()
            suffixes.append(model(**roberta_tensors).pooler_output)

    suffixes = torch.cat(suffixes, dim=0)

    with torch.inference_mode():
        similarities = torch.matmul(prefixes, suffixes.t())
        sorted_scores = torch.sort(similarities, dim=1, descending=True)
        sorted_score_idx, sorted_score_vals = sorted_scores.indices, sorted_scores.values

    ranks = []
    for qnum, (quote_data, context) in tqdm(enumerate(zip(all_masked_quotes, all_contexts)), total=len(all_contexts)):
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
    with open(f"{args.output_dir}/{args.num_examples}instances_{cw[0]}pre_{cw[1]}after.json", 'w') as f:
        json.dump(results, f)


