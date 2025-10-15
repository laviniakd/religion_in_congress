# evaluate reference detection model

import pandas as pd
import spacy
import time
from tqdm import tqdm
import sys
import torch
import transformers

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
def main(CHOSEN_METHOD, TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_GAP, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, EDIT_DISTANCE_THRESHOLD, COSINE_SIM, args, run_id):
    # load data
    verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references(version=BIBLE_VERSION)

    true_positive_df, true_negative_df, cr_df = load_eval_data()
    print("Number of true positive examples: ", len(true_positive_df.index))
    print("Number of true negative examples: ", len(true_negative_df.index))
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
    start_time_all = time.localtime()

    if CHOSEN_METHOD == "fuzzy_string_matching_token":
        fuzzy_string_matching_token_refs = get_fuzzy_string_matching_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), THRESHOLD=EDIT_DISTANCE_THRESHOLD, by_word=True, return_all=True)
        ref_df = convertRefListsToDF(fuzzy_string_matching_token_refs)
        limited_ref_df = ref_df[ref_df['label'] >= EDIT_DISTANCE_THRESHOLD]
    elif CHOSEN_METHOD == "embedding_similarity":
        embedding_similarity_refs = get_embedding_similarity_refs(cr_df, verse_df, args.device, args.batch_size, TOKEN_OVERLAP_THRESHOLD, args, cosine_sim=COSINE_SIM, return_all=True)
        ref_df = convertRefListsToDF(embedding_similarity_refs)
        limited_ref_df = ref_df[ref_df['label'] >= COSINE_SIM]
    elif CHOSEN_METHOD == "ngram_shingling_min_count_overlap":
        ngram_shingling_min_count_overlap_refs = get_ngram_shingling_refs(cr_df, limited_verse_to_citation, list(limited_verse_to_citation.keys()), N=SHINGLE_N, G=SHINGLE_GAP, MIN_NGRAM_MATCH=MIN_NGRAM_MATCH, return_all=True)
        ref_df = convertRefListsToDF(ngram_shingling_min_count_overlap_refs)
        limited_ref_df = ref_df[ref_df['label'] >= MIN_NGRAM_MATCH]
    else:
        print("Invalid method chosen")

    ref_df['method'] = CHOSEN_METHOD
    true_pos_labeled_pos = 0
    true_pos_labeled_neg = 0
    true_neg_labeled_pos = 0
    true_neg_labeled_neg = 0
    

    print("About to count true positives, true negatives, false positives, and false negatives!")
    print("CR_DF: ")
    print(cr_df.head())
    print("REF_DF: ")
    print(ref_df.head())

    not_found = 0
    limited_ref_df_indices = list(limited_ref_df['congress_idx'])
    full_ref_df_indices = list(ref_df['congress_idx'])

    # also note if verse was predicted in row itself (and which one)
    for i, row in cr_df.iterrows():
        if row['congress_idx'] in limited_ref_df_indices:
            row['verse_predicted'] = limited_ref_df[limited_ref_df['congress_idx'] == row['congress_idx']]['verse'].values[0]
            row['score'] = limited_ref_df[limited_ref_df['congress_idx'] == row['congress_idx']]['label'].values[0]
        else:
            row['verse_predicted'] = False
            if row['congress_idx'] in full_ref_df_indices:
                row['score'] = ref_df[ref_df['congress_idx'] == row['congress_idx']]['label'].values[0]
            else:
                print("Row not found in ref_df: ", row['congress_idx'])
                row['score'] = 0.0
                not_found += 1

        # for basic metrics, not attending to whether verse was correctly predicted
        if row['label'] == 1: # will be true_pos_*
            if row['congress_idx'] in list(limited_ref_df['congress_idx']):
                true_pos_labeled_pos += 1
            else:
                true_pos_labeled_neg += 1
        else:
            if row['congress_idx'] in list(limited_ref_df['congress_idx']):
                true_neg_labeled_pos += 1
            else:
                true_neg_labeled_neg += 1

    print("Number of rows not found in ref_df: ", not_found)
    print("This is the number of rows in ref_df: ", len(ref_df.index))

    cr_df.to_csv(args.out_dir + f"references_{CHOSEN_METHOD}.csv", index=False)
    print("CR_DF saved to: ", args.out_dir + f"references_{run_id}.csv")


    print("*** Evaluation metrics ***")
    accuracy = (true_pos_labeled_pos + true_neg_labeled_neg) / (true_pos_labeled_pos + true_pos_labeled_neg + true_neg_labeled_pos + true_neg_labeled_neg)
    if true_pos_labeled_pos + true_neg_labeled_pos == 0:
        precision = 0
    else:
        precision = true_pos_labeled_pos / (true_pos_labeled_pos + true_neg_labeled_pos)
    if true_pos_labeled_pos + true_pos_labeled_neg == 0:
        recall = 0
    else:
        recall = true_pos_labeled_pos / (true_pos_labeled_pos + true_pos_labeled_neg)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    print("F1: ", f1)

    print("*** Number of references for the chosen method ***")
    print(f"{CHOSEN_METHOD}: ", len(ref_df.index))
    if CHOSEN_METHOD == "fuzzy_string_matching_token":
        met = 'fsm'
    elif CHOSEN_METHOD == "embedding_similarity":
        met = 'emb'
    elif CHOSEN_METHOD == "ngram_shingling_min_count_overlap":
        met = 'ngm'
    ref_df.to_csv(args.out_dir + f"{met}/references_{run_id}.csv", index=False)
    print("References saved to: ", args.out_dir + f"{met}/references_{run_id}.csv")
    with open(args.out_dir + f"metrics_{run_id}.txt", "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1: {f1}\n")
    
    with open("/home/laviniad/projects/religion_in_congress/src/references/validation/f1_log.txt", "w") as file:
        file.write(str(f1))

    print("Metrics saved to: ", args.out_dir + f"metrics_{run_id}.txt")
    end_time_all = time.localtime()
    print("Total time: ", time.mktime(end_time_all) - time.mktime(start_time_all))


def load_eval_data():
    TRUE_POSITIVE_PATH = '/home/laviniad/projects/religion_in_congress/src/references/validation/data/true_positive_examples.csv'
    TRUE_NEGATIVE_PATH = '/home/laviniad/projects/religion_in_congress/src/references/validation/data/true_negative_examples.csv'

    true_positive_df = pd.read_csv(TRUE_POSITIVE_PATH)
    true_negative_df = pd.read_csv(TRUE_NEGATIVE_PATH)
    true_negative_df.rename(columns={'speech_id': 'congress_idx', 'sentence': 'text', 'reference': 'verse', 'score': 'label'}, inplace=True)

    true_positive_df['label'] = [1 for i in range(len(true_positive_df.index))]
    true_negative_df['label'] = [0 for i in range(len(true_negative_df.index))]

    # concat
    cr_df = pd.concat([true_positive_df, true_negative_df])
    # remove 'tensor(' and ')' from congress_idx; convert to int
    cr_df['congress_idx'] = cr_df['congress_idx'].apply(lambda x: int(x[7:-1]) if (isinstance(x, str) and 'tensor' in x) else int(x))
    return true_positive_df, true_negative_df, cr_df


def convert_tensor_string_to_int(tensor_string):
    if isinstance(tensor_string, int):
        return tensor_string
    if isinstance(tensor_string, torch.Tensor):
        return int(tensor_string.item())
    
    assert(isinstance(tensor_string, str))
    if 'tensor' not in tensor_string:
        return int(tensor_string)
    return int(tensor_string[7:-1])


def convertRefListsToDF(ref_list):
    def lower_or_return_none(s):
        if (s is None) or (not isinstance(s, str)):
            return 'none'
        return s.lower()

    ref_list = [(convert_tensor_string_to_int(v[0]), v[1], v[2], v[3]) for v in ref_list]
    ref_df = pd.DataFrame(ref_list, columns=['congress_idx', 'text', 'verse', 'label'])
    ref_df['verse'] = ref_df['verse'].apply(lower_or_return_none) # should resolve the case issue
    return ref_df
    

if __name__ == "__main__":
    # options: "fuzzy_string_matching_token", "embedding_similarity", "ngram_shingling_min_count_overlap"
    EDIT_DISTANCE_THRESHOLD = 0.2
    COSINE_SIM = 0.8
    TOKEN_OVERLAP_THRESHOLD = 0.2 # this is used to decide whether to compute cosine similarity between embeddings (expensive) or not
    MIN_SENTENCE_LENGTH = 5
    BIBLE_VERSION = "King James Bible"
    SHINGLE_N = 5
    SHINGLE_G = 1
    MIN_NGRAM_MATCH = 3
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

    if args.id:
        run_id = args.id
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

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
    file_path = f"{args.out_dir}runs/args_{CHOSEN_METHOD}_{run_id}.json"
    with open(file_path, "w") as file:
        json.dump(data, file)

    # Print the file path for reference
    print("Constants and args saved to:", file_path)

    main(CHOSEN_METHOD, TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, SHINGLE_N, SHINGLE_G, BIBLE_VERSION, MIN_NGRAM_MATCH, MIN_NGRAM_MATCH_PCT, EDIT_DISTANCE_THRESHOLD, COSINE_SIM, args, run_id)
    print("Finished run with id:", run_id)
