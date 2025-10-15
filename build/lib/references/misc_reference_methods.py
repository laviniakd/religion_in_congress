import pandas as pd
import spacy
import time
from tqdm import tqdm
import sys
import argparse
import os
import json
import re
import rapidfuzz
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


# verse_citation_dict: text of verse --> citation
def get_fuzzy_string_matching_refs(cr_df, verse_citation_dict, verse_texts, THRESHOLD=80, by_word=False, return_all=False):
    start = time.localtime()
    DEBUG = False
    print("[FUZZY_STRING_MATCHING] Start Time:", time.strftime("%H:%M:%S", start))
    ref_list = []

    verse_to_processed_text = dict(zip(verse_texts, [rapidfuzz.utils.default_process(v) for v in verse_texts]))

    for idx, row in tqdm(cr_df.iterrows()):
        congress_text = row['text']
        congress_idx = row['congress_idx']
        gold = row['verse']

        scorer = distance.Levenshtein.normalized_distance
        verse_text, verse_citation, score = find_best_verse(verse_to_processed_text, verse_citation_dict, congress_text, scorer=scorer, by_word=by_word, thresh=THRESHOLD, gold_standard=gold)
        if DEBUG and (verse_citation == gold):
            print(f"[DEBUG]: Found correct verse for congress text {congress_idx}")
            print(f"[DEBUG]: verse_text is {verse_text}")
            print(f"[DEBUG]: congress_text is {congress_text}")
            print(f"[DEBUG]: verse_citation is {verse_citation}")
            print(f"[DEBUG]: score between texts is {score}")

        if (score >= THRESHOLD) or return_all:
            ref_list.append((row['congress_idx'], congress_text, verse_citation, score))

    end = time.localtime()
    print("[FUZZY_STRING_MATCHING] End Time:", time.strftime("%H:%M:%S", end))
    print("[FUZZY_STRING_MATCHING] Total Time:", time.mktime(end) - time.mktime(start))
    return ref_list


def get_embedding_similarity_refs(cr_df, verse_df, device, batch_size, TOKEN_OVERLAP_THRESHOLD, args, return_all=False, cosine_sim=0.8):
    start = time.localtime()
    print("[EMBEDDING] Start Time:", time.strftime("%H:%M:%S", start))
    ref_list = []

    print(f"Number of potentially religious sentences: {len(cr_df.index)}")

    full_model, model = load_embedding_model(args)
    biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    verseLoader = create_verse_loader(batch_size, verse_df, biTokenizer)
    verse_result_tuples = get_embedded_verses(full_model, model, verseLoader)

    congressLoader = create_congress_loader(batch_size, cr_df, biTokenizer)
    print("Will return all matches, regardless of whether they meet the threshold: ", return_all)
    results = embed_cr_sentences_and_match(device, full_model, model, verse_result_tuples, congressLoader, TOKEN_OVERLAP_THRESHOLD, return_all=return_all, COSINE_SIM_THRESHOLD=cosine_sim)
    ref_list = zip(results['congress_idx'], results['text'], results['verse_citation'], results['cosine_similarity'])

    end = time.localtime()
    print("[EMBEDDING] End Time:", time.strftime("%H:%M:%S", end))
    print("[EMBEDDING] Total Time:", time.mktime(end) - time.mktime(start))

    return ref_list


# threshold controls how much of pos tags need to overlap
def get_syntactic_refs(cr_df, verse_citation_dict, verse_texts, THRESHOLD=0.8):         
    start = time.localtime()
    print("[SYNTACTIC] Start Time:", time.strftime("%H:%M:%S", start))
    ref_list = []

    nlp = spacy.load("en_core_web_sm")

    verse_to_pos = {}
    for verse in verse_texts:
        verse_doc = nlp(verse)
        verse_to_pos[verse] = [(token.text, token.pos_) for token in verse_doc]


    for idx, row in tqdm(cr_df.iterrows()):
        congress_text = row['text']
        congress_idx = row['congress_idx']

        congress_doc = nlp(congress_text)
        congress_pos = [(token.text, token.pos_) for token in congress_doc]

        for c_text, c_pos in congress_pos:
            num_code = {}
            max_i = 0
            for i, pos in enumerate(c_pos):
                if pos not in num_code:
                    num_code[pos] = str(i)
                    max_i = i
            c_pos = ''.join([num_code[pos] for pos in c_pos])

            for verse_sent, v_pos_tuples in verse_to_pos.items():
                for i, (word, pos) in enumerate(v_pos_tuples):
                    if pos not in num_code:
                        num_code[pos] = str(i) if i > max_i else str(max_i + 1)
                        max_i = i if i > max_i else max_i
                # score is normalized token-level levenshtein distance between pos tags
                # transform pos tag sequences into number strings for levenshtein distance
                verse_pos = ''.join([num_code[pos] for _, pos in v_pos_tuples])
                score = distance.Levenshtein.distance(c_pos, verse_pos) / max(len(c_pos), len(verse_pos))

                if score >= THRESHOLD:
                    ref_list.append((congress_idx, c_text, verse_citation_dict[verse], 1 - score))

    end = time.localtime()
    print("[SYNTACTIC] End Time:", time.strftime("%H:%M:%S", end))
    print("[SYNTACTIC] Total Time:", time.mktime(end) - time.mktime(start))

    return ref_list


# use ngram shingling w bible verses to find references
# N = n-gram order
# G = minimum gap of skip n-grams
# U = maximum distinct series in the posting list (more than U-choose-2 pairs in posting list = discard)
# theoretically the number of n-grams required to be certain about an exact match is len(verse) - N + 1; 
# however, flexibility wrt inexact quotations means we want a lower threshold. for now, we set this explicitly
# MIN_NGRAM_MATCH = minimum number of n-grams that must match to consider a verse a match
# verse_citation_dict is a dictionary mapping verse text to citation
def get_ngram_shingling_refs(cr_df, verse_citation_dict, verse_texts, N=5, G=1, U=10000, MIN_NGRAM_MATCH=5, return_all=False):
    start = time.localtime()
    print("[SHINGLING] Start Time:", time.strftime("%H:%M:%S", start))
    # gets "shingles" for given text and n -- all n-grams in the text
    # assumes text is tokenized appropriately
    def create_ngram_shingles(text, n, g):
        text = [t.lower() for t in text]
        ngrams = []
        i = 0

        while i < len(text) - n + 1:
            ngram = '_'.join(text[i:i+n])
            ngrams.append(ngram)
            i += g

        return ngrams # creates (hashable) single strings from n-grams

    ref_list = []

    # get shingles for verses
    print("Creating ngram shingles for Bible verses...")
    verse_shingles = {}
    for verse in verse_texts:
        # is verse text --> shingles
        verse_shingles[verse] = create_ngram_shingles(nltk.word_tokenize(verse), N, G)

    # shingle each congress text and find best match
    print("Creating ngram shingles for Congress texts...")
    congress_shingle_data = []

    #print("[DEBUG] Length of cr_df: ", len(cr_df))
    for _, row in tqdm(cr_df.iterrows()):
        congress_text = row['text']
        congress_idx = row['congress_idx']

        congress_shingles = create_ngram_shingles(nltk.word_tokenize(congress_text), N, G)
        congress_shingle_data.append((congress_idx, congress_text, congress_shingles))
    #print("[DEBUG] Length of congress_shingle_data: ", len(congress_shingle_data))

    # create ngram index
    ngram_to_doc = {}
    ngram_to_verse = {}

    print("Creating ngram index of Bible verses and Congress texts...")
    # ids are citations
    print("Iterating through verses")
    for verse, shingle in verse_shingles.items():
        for n_gram in shingle:
            if n_gram not in ngram_to_doc:
                ngram_to_doc[n_gram] = []
            ngram_to_doc[n_gram].append('verse:' + verse)
            if n_gram not in ngram_to_verse:
                ngram_to_verse[n_gram] = []
            ngram_to_verse[n_gram].append(verse)

    # ids are numbers
    print("Iterating through congress texts")
    for doc, text, shingle in congress_shingle_data:
        for n_gram in shingle:
            if n_gram not in ngram_to_doc:
                ngram_to_doc[n_gram] = []
            ngram_to_doc[n_gram].append('speech:' + str(doc))


    # discard ngrams that only appear once
    temp = ngram_to_doc.copy()
    for gram, docs in temp.items():
        if len(docs) == 1:
            del ngram_to_doc[gram]
        if len(docs) > U:
            del ngram_to_doc[gram]

    
    # find best match for each congress text
    # following Viral Texts, we require at least MIN_NGRAM_MATCH shingles to match
    if MIN_NGRAM_MATCH < 1:
        print("Assuming MIN_NGRAM_MATCH < 1 means we want to match a percentage of shingles")
        MATCHING_PERCENTAGE_OF_SHINGLES = True
    else:
        print("Assuming MIN_NGRAM_MATCH >= 1 means we want to match a minimum number of shingles")
        MATCHING_PERCENTAGE_OF_SHINGLES = False

    print(f"Finding best match for each congress text; requiring at least {MIN_NGRAM_MATCH} shingles to match")
    for doc, text, shingle in congress_shingle_data:
        best_score = 0
        best_verse = None
        best_verse_citation = None
        shingles = set([s for s in shingle if s in ngram_to_doc.keys()])

        if len(shingles) > 0:
            for verse, ngrams in verse_shingles.items():
                single_verse_shingle_set = set([s for s in ngrams if s in ngram_to_doc.keys()])
                score = len(shingles.intersection(single_verse_shingle_set))
                if MATCHING_PERCENTAGE_OF_SHINGLES:
                    score = score / len(shingles)
                    if (score >= MIN_NGRAM_MATCH or return_all) and (score > best_score):
                        best_score = score
                        best_verse = verse
                        best_verse_citation = verse_citation_dict[verse] # assuming is verse text --> citation
                elif (score >= MIN_NGRAM_MATCH or return_all) and (score > best_score):
                    best_score = score
                    best_verse = verse
                    best_verse_citation = verse_citation_dict[verse] # assuming is verse text --> citation

            if best_verse is not None:
                ref_list.append((doc, text, best_verse_citation, best_score)) 
                # note that score is a count depending on whether we are matching 
                # a percentage of shingles or a minimum number of shingles
            else:
                if return_all:
                    ref_list.append((doc, text, 'None', 0.0))
        else:
            if return_all:
                ref_list.append((doc, text, 'None', 0.0))

    end = time.localtime()
    print("[SHINGLING] End Time:", time.strftime("%H:%M:%S", end))
    print("[SHINGLING] Total Time:", time.mktime(end) - time.mktime(start))

    return ref_list


def get_memetracker_refs(cr_df, verse_citation_dict, verse_texts, run_id, use_all_verses=True):
    start = time.localtime()
    print("[MEMETRACKER] Start Time:", time.strftime("%H:%M:%S", start))
    ref_list = []

    phrases = list(verse_texts) + list(cr_df['text'])
    connected_components = find_connected_components(phrases, delta=1, k=10, out_path=f"/data/laviniad/sermons-ir/references/partitions_{run_id}.txt")

    # find connected components with verse texts in them
    def is_verse_component(component):
        for verse in verse_texts:
            if verse in component:
                return True
        return False
    verse_connected_components = [c for c in connected_components if is_verse_component(c)]

    # add to ref_list
    for idx, row in cr_df.iterrows():
        for verse_component in verse_connected_components:
            if row['text'] in verse_component:
                ref_list.append((row['congress_idx'], row['text'], verse_citation_dict[row['text']], 1.0))

    end = time.localtime()
    print("[MEMETRACKER] End Time:", time.strftime("%H:%M:%S", end))
    print("[MEMETRACKER] Total Time:", time.mktime(end) - time.mktime(start))
    return ref_list