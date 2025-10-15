# Collect Biblical references using several methods:
# - citation recognition via regexes 
# - fuzzy string matching w Indel
# - embedding similarity w SBERT
# - ngram shingling
# - fuzzy string matching w Levenshtein distance

# from meeting notes:
#Could use a bunch of methods, randomly sample returned references, and estimate precision from that (after doing human annotation of some kind)
#E.g., citations + fuzzy string matching + SBERT embedding + Viral Texts + Levenshtein distance + etc.
#Paraphrase/quotation detection seems like the right scope
#Evaluate along both is the task similar and does the method make sense
#For pair annotation a couple hundred should make sense

# load imports
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

from src.references.misc_reference_methods import get_fuzzy_string_matching_refs, get_embedding_similarity_refs, get_ngram_shingling_refs, get_syntactic_refs, get_memetracker_refs


# args should contain the following:
# - model_dir
# - input
# - out_dir
# - congress_errata_path
# - device
# - keyword_fraction_threshold
# - batch_size
# - sample
# - debug
def main(TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_GAP, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, args, run_id):
    # load data
    verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references(version=BIBLE_VERSION)
    cr_df = congress_utils.load_full_df_from_raw(args.input)

    filtered_df = load_and_filter_cr(args, cr_df, KEYWORD_FILTER_THRESHOLD=1, EMBED_ALL=True) # not filtering by KW freq
    cr_df = create_inference_df(MIN_SENTENCE_LENGTH, filtered_df)
    cr_df = cr_df.sample(args.sample) if args.sample > 0 else cr_df

    print("Hyperparameters...")
    print(f"MIN_SENTENCE_LENGTH (filtering): {MIN_SENTENCE_LENGTH}")
    print(f"Sample size (filtering): {args.sample}")

    print(f"TOKEN_OVERLAP_THRESHOLD (used to shortcut cosine similarity in embedding method): {TOKEN_OVERLAP_THRESHOLD}")
    print(f"SHINGLE_N (n-gram order for shingling): {SHINGLE_N}")
    print(f"SHINGLE_GAP (minimum gap between n-grams for shingling): {SHINGLE_GAP}")


    # for each method, iterate through a subset of the cr data and collect references
    # references are recorded in one format for each method: (speech_id, sentence, reference)
    print("--- Collecting references using different methods ---")
    start_time_all = time.localtime()
    fuzzy_string_matching_token_refs = get_fuzzy_string_matching_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), THRESHOLD=80, by_word=True)
    fuzzy_string_matching_character_refs = get_fuzzy_string_matching_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), THRESHOLD=60, by_word=False)
    embedding_similarity_refs = get_embedding_similarity_refs(cr_df, verse_df, args.device, args.batch_size, TOKEN_OVERLAP_THRESHOLD, args)
    ngram_shingling_min_count_overlap_refs = get_ngram_shingling_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), N=SHINGLE_N, G=SHINGLE_GAP, MIN_NGRAM_MATCH=MIN_NGRAM_MATCH)
    ngram_shingling_min_perc_overlap_refs = get_ngram_shingling_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), N=SHINGLE_N, G=SHINGLE_GAP, MIN_NGRAM_MATCH=MIN_NGRAM_MATCH_PCT)
    #syntactic_refs = get_syntactic_refs(cr_df, limited_verse_to_citation, verse_df['text'])
    memetracker_refs = get_memetracker_refs(cr_df, limited_verse_to_citation, verse_df['text'], run_id)
    end_time_all = time.localtime()
    print("Total time for all methods:", time.mktime(end_time_all) - time.mktime(start_time_all))
    
    fuzzy_string_matching_token_df = convertRefListsToDF(fuzzy_string_matching_token_refs)
    fuzzy_string_matching_token_df['method'] = 'fsm_levenshtein_token'

    fuzzy_string_matching_character_df = convertRefListsToDF(fuzzy_string_matching_character_refs)
    fuzzy_string_matching_character_df['method'] = 'fsm_levenshtein_char'

    embedding_similarity_df = convertRefListsToDF(embedding_similarity_refs)
    embedding_similarity_df['method'] = 'embedding_similarity'

    ngram_shingling_min_count_overlap_df = convertRefListsToDF(ngram_shingling_min_count_overlap_refs)
    ngram_shingling_min_count_overlap_df['method'] = 'ngram_shingling_count'

    ngram_shingling_min_perc_overlap_df = convertRefListsToDF(ngram_shingling_min_perc_overlap_refs)
    ngram_shingling_min_perc_overlap_df['method'] = 'ngram_shingling_perc'

    #syntactic_df = convertRefListsToDF(syntactic_refs)
    #syntactic_df['method'] = 'syntactic'

    memetracker_df = convertRefListsToDF(memetracker_refs)
    memetracker_df['method'] = 'memetracker'

    # save the references and their labels
    ref_df = pd.concat([fuzzy_string_matching_token_df, fuzzy_string_matching_character_df, embedding_similarity_df, 
        ngram_shingling_min_count_overlap_df, ngram_shingling_min_perc_overlap_df, memetracker_df])
    print("*** Number of references for each method ***")
    print("Fuzzy string matching (token): ", len(fuzzy_string_matching_token_df.index))
    print("Fuzzy string matching (character): ", len(fuzzy_string_matching_character_df.index))
    print("Embedding similarity: ", len(embedding_similarity_df.index))
    print("Ngram shingling (min count overlap): ", len(ngram_shingling_min_count_overlap_df.index))
    print("Ngram shingling (min perc overlap): ", len(ngram_shingling_min_perc_overlap_df.index))
    #print("Syntactic: ", len(syntactic_df.index))
    print("Memetracker: ", len(memetracker_df.index))

    print("--> Number of (non-unique) references found, across all methods: ", len(ref_df.index))
    ref_df.to_csv(args.out_dir + f"references_{run_id}.csv", index=False)
    print("References saved to:", args.out_dir + f"references_{run_id}.csv")

def convertRefListsToDF(ref_list):
    ref_list = [(v[0], v[1], v[2], v[3]) for v in ref_list]
    ref_df = pd.DataFrame(ref_list, columns=['speech_id', 'sentence', 'reference', 'score'])
    return ref_df
    

if __name__ == "__main__":
    TOKEN_OVERLAP_THRESHOLD = 0.2 # this is used to decide whether to compute cosine similarity between embeddings (expensive) or not
    MIN_SENTENCE_LENGTH = 5
    BIBLE_VERSION = "King James Bible"
    SHINGLE_N = 5
    SHINGLE_G = 2
    MIN_NGRAM_MATCH = 5
    MIN_NGRAM_MATCH_PCT = 0.8
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
    # this predates decidingn not to use tuned mpnet
    parser.add_argument('--model_dir', default="sentence-transformers/all-mpnet-base-v2", type=str)
    parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/references/", type=str)
    parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()

    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Create a dictionary to store constants and args
    data = {
        "TOKEN_OVERLAP_THRESHOLD": TOKEN_OVERLAP_THRESHOLD,
        "MIN_SENTENCE_LENGTH": MIN_SENTENCE_LENGTH,
        "BIBLE_VERSION": BIBLE_VERSION,
        "SHINGLE_N": SHINGLE_N,
        "SHINGLE_G": SHINGLE_G,
        "MIN_NGRAM_MATCH": MIN_NGRAM_MATCH,
        "MIN_NGRAM_MATCH_PCT": MIN_NGRAM_MATCH_PCT,
        "args": vars(args)
    }

    # Save the data to a file with the run id
    file_path = f"{args.out_dir}runs/args_{run_id}.json"
    with open(file_path, "w") as file:
        json.dump(data, file)

    # Print the file path for reference
    print("Constants and args saved to:", file_path)

    main(TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_G, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, args, run_id)
    print("Finished run with id: ", run_id)
