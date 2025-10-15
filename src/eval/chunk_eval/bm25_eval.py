"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Evaluate BM-25's performance on the sermon dataset
- with limited context (N previous and M following sentences)
- with easy and easiest settings (using only most popular verses and 
  using only verses that appear in the sermon) -- easiest NYI
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import argparse
from tqdm import tqdm
import json
from src.bm25 import BM25Okapi
from data.bible_utils import bible_helper, build_candidates
from data.data_utils import get_verse_dicts
from src.eval.eval_utils import load_paired_data

from src.relic_utils.utils import build_lit_instance, print_results, NUM_SENTS

COMMENTARY_VERSE = '/data/laviniad/sermons-ir/commentary-verse-dfs/'
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/AmericanKJVBible.txt'
FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'
POPULAR_VERSES = '/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--split', default="train", type=str)
parser.add_argument('--output_dir', default="/home/laviniad/projects/religion_in_congress/results/commentary_verse/bm25/", type=str)
parser.add_argument('--num_examples', default=100, type=int)
parser.add_argument('--easy', help='only use most popular set of verses as target', action='store_true')
parser.add_argument('--easy_verse_number', default=250, type=int, help='number of popular verses in easy setting')
#parser.add_argument('--easiest', description='only use verses mentioned in sermon as target', action='store_true')
args = parser.parse_args()

bible_df = bible_helper(BIBLE_PATH)
verse_text_dict, fuzzy_citation_dict = get_verse_dicts(FUZZY_CITATION_DICT_PATH, BIBLE_PATH)
verse_text_dict = {k: v.lower() for k,v in verse_text_dict.items()}

text_verse_dict = {v: k for k,v in verse_text_dict.items()}

context_windows = [(1,1), (2,2), (3,3), (4,4),
                   (0,1), (1,0), (2,0), (0,2),
                   (0,3), (3,0), (4,0), (0,4)]

total = 0
submission_data = {}


cands = build_candidates(bible_df)
verse_to_idx = {v.lower(): k for k,v in enumerate(cands)}
verse_text_dict = {k: v.lower() for k,v in verse_text_dict.items()}
text_verse_dict = {v: k for k,v in verse_text_dict.items()}

tokenized_corpus = [verse.split(" ") for verse in cands]
bm25 = BM25Okapi(tokenized_corpus)

# handle 'easy' setting
pop_verses = None
if args.easy:
    popular_verses = pd.read_csv(POPULAR_VERSES)
    pop_verses = list(popular_verses[:args.easy_verse_number]['verse']) # process popular verses into list
    cands = [v for v in cands if text_verse_dict[v.lower()] in pop_verses] # limit candidates to those verses
    verse_to_idx = {v.lower(): k for k,v in enumerate(cands)}
    pop_bm25 = BM25Okapi([verse.split(" ") for verse in cands])


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
                                                                            args.num_examples, easy=args.easy,
                                                                           pop_verses = pop_verses)
    full_len = float(len(commentary_verse_df.index))

    ranks = []
    for qnum, (quote_data, context) in tqdm(enumerate(zip(all_masked_quotes, all_contexts)), total=len(all_contexts)):
        quote_id, verse = quote_data[0], quote_data[1]

        tokenized_query = (context[0] + context[1]).split(" ")  # just concatenating prefix and suffix
        if args.easy:
            result_verse_scores = pop_bm25.get_scores(tokenized_query)
        else:
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
    with open(f"{args.output_dir}/{args.num_examples}instances_{cw[0]}pre_{cw[1]}after.json", 'w') as f:
        json.dump(results, f)

