from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

import pandas as pd
import argparse
from tqdm import tqdm
import json
from src.bm25 import BM25Okapi
from data.bible_utils import bible_helper, build_candidates
from data.data_utils import get_verse_dicts
from src.eval.eval_utils import load_whole_sermon_data

from src.relic_utils.utils import build_lit_instance, print_results, NUM_SENTS

SERMON_DF_PATH = '/data/laviniad/sermons-ir/sermoncentral/fsm/'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'

parser = argparse.ArgumentParser()
parser.add_argument('--split', default="train", type=str)
parser.add_argument('--output_dir', default="/home/laviniad/projects/religion_in_congress/results/whole_sermon/colbert/", type=str)
parser.add_argument('--reindex', action='store_true')
parser.add_argument('--formatted_data_path', default="/data/laviniad/sermons-ir/whole-sermon-dfs/fsm/colbert/", type=str)
parser.add_argument('--num_examples', default=100, type=int)
args = parser.parse_args()

bible_df = bible_helper(BIBLE_PATH)
verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
verse_text_dict = {k: v.lower() for k,v in verse_text_dict.items()}

text_verse_dict = {v: k for k,v in verse_text_dict.items()}

total = 0
submission_data = {}


cands = build_candidates(bible_df)
verse_to_idx = {v.lower(): k for k,v in enumerate(cands)}

config = ColBERTConfig(nbits=2, root="/path/to/experiments")

if args.reindex:
    indexer = Indexer(checkpoint="/path/to/checkpoint", config=config)
    indexer.index(name="msmarco.nbits=2", collection=)

searcher = Searcher(index="msmarco.nbits=2", config=config)
queries = Queries("/path/to/MSMARCO/queries.dev.small.tsv")
ranking = searcher.search_all(queries, k=100)
ranking.save("msmarco.nbits=2.ranking.tsv")


print(f"{'-' * 25} Now evaluating {'-' * 25}")

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

sermon_df, sermon_verse_data, sermon_texts = load_whole_sermon_data(SERMON_DF_PATH, args.split, verse_text_dict, verse_to_idx, args.num_examples, fuzzy_citation_dict)
full_len = float(len(commentary_verse_df.index))

ranks = []

for qnum, (quote_data, sermon) in tqdm(enumerate(zip(sermon_verse_lists, sermon_texts)), total=len(sermon_texts)):
    quote_id, verse = quote_data[0], quote_data[1]

    tokenized_query = (context[0] + context[1]).split(" ")  # just concatenating prefix and suffix
    result_verse_scores = bm25.get_scores(tokenized_query)
    sorted_scores = dict(sorted(enumerate(result_verse_scores), key=lambda x: x[0], reverse=True))
    sorted_score_idx, sorted_score_vals = list(sorted_scores.keys()), list(sorted_scores.values())

    gold_rank = sorted_score_idx.index(quote_id) + 1
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
    print(f"Printed results to {f.name}")


