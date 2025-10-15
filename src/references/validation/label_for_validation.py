# sampling a bunch of random sentences and labeling with both embedding and n-gram scores

import pandas as pd
import spacy
import time
from tqdm import tqdm
import sys
import argparse
import os
import json
import re
from rapidfuzz import distance
import nltk

from data import congress_utils, data_utils
from src.references.train_biencoder import BiModel
from src.references.fsm_retrieve_and_infer import find_best_verse
from src.references.retrieve_and_infer import load_embedding_model, load_bible_data_for_references, load_and_filter_cr, create_inference_df, get_embedded_verses, create_verse_loader, create_congress_loader, embed_cr_sentences_and_match
from src.references.memetracker_utils import find_connected_components

from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, dataloader
import torch
import datetime

from src.references.misc_reference_methods import get_fuzzy_string_matching_refs, get_embedding_similarity_refs, get_ngram_shingling_refs
from src.references.validation import eval

def main(CHOSEN_METHOD, TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_GAP, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, EDIT_DISTANCE_THRESHOLD, COSINE_SIM, args):
    verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references(version=BIBLE_VERSION)
    SAMPLE = 1000000

    cr_df = pd.read_csv(f"/data/laviniad/sermons-ir/references/test/cr_df_{SAMPLE}.csv")
    print("Number of rows in cr_df: ", len(cr_df.index))

    print("Types of columns in cr_df: ", cr_df.dtypes)

    print("*** Hyperparameters... ***")
    print(f"MIN_SENTENCE_LENGTH (filtering): {MIN_SENTENCE_LENGTH}")
    print(f"Sample size (filtering): {args.sample}")

    print(f"TOKEN_OVERLAP_THRESHOLD (used to shortcut cosine similarity in embedding method): {TOKEN_OVERLAP_THRESHOLD}")
    print(f"SHINGLE_N (n-gram order for shingling): {SHINGLE_N}")
    print(f"SHINGLE_GAP (minimum gap between n-grams for shingling): {SHINGLE_GAP}")

    # for each method, iterate through a subset of the cr data and collect references
    # references are recorded in one format for each method: (speech_id, sentence, reference)
    print("--- Collecting references using different methods ---")

    if CHOSEN_METHOD == "fuzzy_string_matching_token":
        fuzzy_string_matching_token_refs = get_fuzzy_string_matching_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), THRESHOLD=EDIT_DISTANCE_THRESHOLD, by_word=True, return_all=True)
        ref_df = eval.convertRefListsToDF(fuzzy_string_matching_token_refs)
        limited_ref_df = ref_df[ref_df['label'] >= EDIT_DISTANCE_THRESHOLD]
    elif CHOSEN_METHOD == "embedding_similarity":
        embedding_similarity_refs = get_embedding_similarity_refs(cr_df, verse_df, args.device, args.batch_size, TOKEN_OVERLAP_THRESHOLD, args, cosine_sim=COSINE_SIM, return_all=True)
        ref_df = eval.convertRefListsToDF(embedding_similarity_refs)
        limited_ref_df = ref_df[ref_df['label'] >= COSINE_SIM]
    elif CHOSEN_METHOD == "ngram_shingling_min_count_overlap":
        ngram_shingling_min_count_overlap_refs = get_ngram_shingling_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), N=SHINGLE_N, G=SHINGLE_GAP, MIN_NGRAM_MATCH=MIN_NGRAM_MATCH, return_all=True)
        ref_df = eval.convertRefListsToDF(ngram_shingling_min_count_overlap_refs)
        limited_ref_df = ref_df[ref_df['label'] >= MIN_NGRAM_MATCH]
    else:
        print("Invalid method chosen")

    ref_df['method'] = CHOSEN_METHOD
    print("CR_DF: ")
    print(cr_df.head())
    print("REF_DF: ")
    print(ref_df.head())

    not_found = 0
    limited_ref_df_indices = list(limited_ref_df['congress_idx'])

    # also note if verse was predicted in row itself (and which one)
    ref_df['positive_match_with_hyperparam'] = ref_df['congress_idx'].apply(lambda x: x in limited_ref_df_indices)

    ref_df.to_csv(args.out_dir + f"references_{CHOSEN_METHOD}_SAMPLE.csv", index=False)
    print("CR_DF saved to: ", args.out_dir + f"references_{CHOSEN_METHOD}_SAMPLE.csv")


if __name__ == "__main__":
    # options: "fuzzy_string_matching_token", "embedding_similarity", "ngram_shingling_min_count_overlap"
    EDIT_DISTANCE_THRESHOLD = 0.2
    COSINE_SIM = 0.5
    TOKEN_OVERLAP_THRESHOLD = 0.2 # this is used to decide whether to compute cosine similarity between embeddings (expensive) or not
    MIN_SENTENCE_LENGTH = 5
    BIBLE_VERSION = "King James Bible"
    SHINGLE_N = 3
    SHINGLE_G = 1
    MIN_NGRAM_MATCH = 4
    MIN_NGRAM_MATCH_PCT = 0.8
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
    # this predates decidingn not to use tuned mpnet
    parser.add_argument('--model_dir', default="sentence-transformers/all-mpnet-base-v2", type=str)
    parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/references/", type=str)
    parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--id', type=str)
    parser.add_argument('--method', default='ngram_shingling_min_count_overlap')
    parser.add_argument('--hyperparam', default=0.2)
    args = parser.parse_args()

    if args.method == "fuzzy_string_matching_token":
        CHOSEN_METHOD = "fuzzy_string_matching_token"
        EDIT_DISTANCE_THRESHOLD = float(args.hyperparam)
    elif args.method == "embedding_similarity":
        CHOSEN_METHOD = "embedding_similarity"
        COSINE_SIM = float(args.hyperparam)
    elif args.method == "ngram_shingling_min_count_overlap":
        CHOSEN_METHOD = "ngram_shingling_min_count_overlap"
        SHINGLE_N = int(args.hyperparam)

    data = {
        "TOKEN_OVERLAP_THRESHOLD": TOKEN_OVERLAP_THRESHOLD,
        "MIN_SENTENCE_LENGTH": MIN_SENTENCE_LENGTH,
        "BIBLE_VERSION": BIBLE_VERSION,
        "SHINGLE_N": SHINGLE_N,
        "SHINGLE_G": SHINGLE_G,
        "MIN_NGRAM_MATCH": MIN_NGRAM_MATCH,
        "MIN_NGRAM_MATCH_PCT": MIN_NGRAM_MATCH_PCT,
        "EDIT_DISTANCE_THRESHOLD": EDIT_DISTANCE_THRESHOLD,
        "COSINE_SIM": COSINE_SIM,
        "args": vars(args)
    }

    # Save the data to a file with the run id
    file_path = f"{args.out_dir}runs/args_{CHOSEN_METHOD}_SAMPLE.json"
    with open(file_path, "w") as file:
        json.dump(data, file)

    # Print the file path for reference
    print("Constants and args saved to:", file_path)

    main(CHOSEN_METHOD, TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_G, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, EDIT_DISTANCE_THRESHOLD, COSINE_SIM, args)
