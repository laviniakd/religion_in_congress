# in general: sample random negative examples from the CR data, assuming they are not verse references; 
# then pick top verse match according to all three methods
# methods: 
# imports
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
from src.references.validation.collect_refs_using_different_methods import convertRefListsToDF

from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, dataloader
import torch
import datetime

from src.references.misc_reference_methods import get_fuzzy_string_matching_refs, get_embedding_similarity_refs, get_ngram_shingling_refs

BIBLE_VERSION = 'King James Bible'
OUT_DATA_PATH = '/home/laviniad/projects/religion_in_congress/src/references/validation/data/true_negative_examples.csv'
MIN_SENTENCE_LENGTH = 5
SHINGLE_N = 4
SHINGLE_GAP = 2
MIN_NGRAM_MATCH = 3
DEVICE = 'cuda:0'
BATCH_SIZE = 64
TOKEN_OVERLAP_THRESHOLD = 0.2


def main(args):
    # load bible data
    verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references(version=BIBLE_VERSION)
    cr_df = congress_utils.load_full_df_from_raw('/data/corpora/congressional-record/')

    # transform into sentence-wise df
    cr_df = create_inference_df(MIN_SENTENCE_LENGTH, cr_df)

    # sample 2000 for each method
    cr_df = cr_df.sample(6000)
    # split into 3 dfs
    cr_df_1 = cr_df[:2000]
    cr_df_2 = cr_df[2000:4000]
    cr_df_3 = cr_df[4000:]

    # get references for each method
    print("Collecting references using different methods...")
    refs_1 = get_ngram_shingling_refs(cr_df_1, limited_verse_to_citation, list(limited_verse_to_citation.keys()), N=SHINGLE_N, G=SHINGLE_GAP, MIN_NGRAM_MATCH=MIN_NGRAM_MATCH, return_all=True)
    refs_2 = get_fuzzy_string_matching_refs(cr_df_2, limited_verse_to_citation, list(limited_verse_to_citation.keys()), THRESHOLD=80, by_word=True, return_all=True)
    refs_3 = get_embedding_similarity_refs(cr_df_3, verse_df, DEVICE, BATCH_SIZE, TOKEN_OVERLAP_THRESHOLD, args, return_all=True)
    refs_1 = convertRefListsToDF(refs_1)
    refs_2 = convertRefListsToDF(refs_2)
    refs_3 = convertRefListsToDF(refs_3)

    # merge into one df
    strong_negative_examples = pd.concat([refs_1, refs_2, refs_3])
    strong_negative_examples.to_csv(OUT_DATA_PATH, index=False)
    print("Dumped examples to ", OUT_DATA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
    # this predates decidingn not to use tuned mpnet
    parser.add_argument('--model_dir', default="sentence-transformers/all-mpnet-base-v2", type=str)
    parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/references/", type=str)
    parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--batch_size', default=64, type=float)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    main(args)
