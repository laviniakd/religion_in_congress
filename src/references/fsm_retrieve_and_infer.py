import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import gc

import rapidfuzz
from rapidfuzz import fuzz, process
from data.bible_utils import comp_bible_helper
from data.congress_utils import induce_party_and_state
from data.data_utils import get_lexical_overlap
from src.references.retrieve_and_infer import load_bible_data_for_references

import numpy as np

from datasets import Dataset
import nltk
import torch

tqdm.pandas()

from data import congress_utils
from joblib import Parallel, delayed

MIN_SENTENCE_LENGTH = 5
N_JOBS = 8

STOPS = set(nltk.corpus.stopwords.words('english'))

def process_text_for_fuzzy_comparison(text):
    text = nltk.word_tokenize(text)
    # remove stop words
    text = [word for word in text if word.lower() not in STOPS]
    text = " ".join(text)
    return text

# verse_citations is verse text -> citation
def find_best_verse(verse_texts, verse_citations, congress_text, scorer=fuzz.ratio, by_word=False, thresh=None, gold_standard=None):
    DEBUG = False

    congress_text = rapidfuzz.utils.default_process(congress_text)

    if DEBUG:
        print("Congress text: ")
        print(congress_text)
        if gold_standard:
            print("Gold standard verse: ")
            gold_standard = rapidfuzz.utils.default_process(gold_standard)
            print(gold_standard)
            print("Score: ")
            print(scorer(process_text_for_fuzzy_comparison(congress_text), process_text_for_fuzzy_comparison(gold_standard)))

    best_verse = None
    best_score = 0
    best_citation = None

    for v_text, v_processed in verse_texts.items():
        citation = verse_citations[v_text]
        #v_text = rapidfuzz.utils.default_process(v_text)
        # "default_process" lowercases, removes punctuation and other nonalphanumeric characters, and trims whitespace
        if by_word:
            if thresh:
                score = scorer(process_text_for_fuzzy_comparison(congress_text), process_text_for_fuzzy_comparison(v_processed))
            else:
                score = scorer(process_text_for_fuzzy_comparison(congress_text), process_text_for_fuzzy_comparison(v_processed))
        else:
            if thresh:
                score = scorer(congress_text, v_text, score_cutoff=thresh)
            else:
                score = scorer(congress_text, v_text)
        
        score = 1 - score # (now higher is better)
        if score > best_score:
            best_score = score
            best_verse = v_text
            best_citation = citation

    return best_verse, best_citation, best_score

def main(MIN_SENTENCE_LENGTH, args):
# load congressional data
    print("Loading congressional data")
    congressional_df = congress_utils.load_full_df_from_raw(args.input, remove_procedural_speeches=True)
    congressional_df = induce_party_and_state(congressional_df)

    if args.debug:
        congressional_df = congressional_df.sample(1000)
        print("In debug mode; sampled 10000 documents from congressional record")
    elif args.sample > 0:
        congressional_df = congressional_df.sample(args.sample)
        print(f"Due to sample being passed, sampled down to {args.sample} instances")
    else:
        print("Did not sample, since argument was not passed; using full CR")

# retrieve the speeches that contain these keywords and create new df with rows containing sentence + verse + sermon idx of sentence
    print("Filtering congressional data by keywords")
    def filter(speech, threshold):
        overlap, _, _, _ = get_lexical_overlap(speech)
        return overlap > threshold

# adaptated from src/references/retrieve_and_infer.py, so working around those variable names a bit
    filtered_df = congressional_df

    print(f"Number of religious speeches: {len(filtered_df.index)}, or {100 * (len(filtered_df.index) / len(congressional_df.index))}% of CR")

    infer_df = []
    print("Splitting speeches into sentences")
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df.index)):
        sentence_list = nltk.sent_tokenize(row['text'])
        for i,s in enumerate(sentence_list):
            if len(s.split()) >= MIN_SENTENCE_LENGTH:
                infer_df.append({
                'congress_idx': idx,
                'text': s
            })

    infer_df = pd.DataFrame(infer_df)
    print(f"Number of potentially religious sentences: {len(infer_df.index)}")

# load verses
    print("Loading verses...")
    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references()

# embed congress sentences
    print("Creating Congress dataset obj")
    congressDataset = Dataset.from_pandas(infer_df)
    print("Creating Bible verse dataset obj")
    verseDataset = Dataset.from_pandas(verse_df)

# create new df of references given the above pairs
    result_df = []
    print("Finding most similar Bible verses")
    print("Retrieving result matrix...")

    for row in tqdm(congressDataset):
        congress_text = row['text']
        congress_idx = row['congress_idx']

    ## using fuzz.ratio, which is the Indel distance -- number of insertions and deletions required to get one string from other -- normed to between 0 and 100
        verse_text, verse_citation, _ = find_best_verse(verseDataset['text'], verseDataset['citation'], congress_text)
    
        result_df.append({
        'congress_idx': congress_idx,
        'text': congress_text,
        'most_similar_verse': verse_text,
        'verse_citation': verse_citation,
    })

    result_df = pd.DataFrame(result_df)

# dump
    if not args.debug:
        if args.sample < 0:
            result_df.to_csv(args.out_dir + 'results.csv')
            print(f"Dumped result_df to {args.out_dir}results.csv")
        else:
            result_df.to_csv(args.out_dir + f'sample{args.sample}_results.csv')
            print(f"Dumped result_df to {args.out_dir}sample{args.sample}_results.csv")
    else:
        print("In debug mode; did not dump results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
    parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/modeling/fsm_results/", type=str)
    parser.add_argument('--congress_errata_path', default="/data/laviniad/congress_errata/", type=str)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(MIN_SENTENCE_LENGTH, args)
