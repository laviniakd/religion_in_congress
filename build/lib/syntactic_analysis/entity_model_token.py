# for the input token, record
# - whether god is mentioned as subject or object
# - attached verb (if god is a subject) + whether present or past
# - common adjectives
# - possessives

import spacy
from tqdm import tqdm
import pandas as pd
from data import congress_utils, data_utils
import numpy as np
import json
import time
from random import randint

# constants
OUTPUT_PATH = "/data/laviniad/sermons-ir/modeling_god/"

TOKEN_OF_INTEREST = "god"
RUN_ID = str(randint(100, 999)) # random 3-digit
INPUT = "/data/corpora/congressional-record/"
EXPAND_TO_SYNONYMS = False
DEBUG = False

if EXPAND_TO_SYNONYMS:
    TOKEN_OF_INTEREST = data_utils.get_synonyms(TOKEN_OF_INTEREST)

print("Run ID: ", RUN_ID)
print("Token of interest: ", TOKEN_OF_INTEREST)
print("Now dumping args")

with open(OUTPUT_PATH + 'args.json', 'w') as f:
    json.dump({'token_of_interest': TOKEN_OF_INTEREST,
               'run_id': RUN_ID,
               'input_root': INPUT,
               'expand_to_synonyms': EXPAND_TO_SYNONYMS,
                'debug': DEBUG
               }, f)
    

# important functions for extracting spacy data
def isinstance_of_interest(token):
    if EXPAND_TO_SYNONYMS:
        return token.text.lower() in TOKEN_OF_INTEREST
    return token.text.lower() == TOKEN_OF_INTEREST

def get_verb_data(token):
    verb, verb_tense, verb_obj_list, verb_subj = None, None, None, None
    
    # token of interest is subject
    if token.dep_ in ["nsubj", "nsubjpass"]:
        verb = token.head
        verb_tense = 'present' if verb.tag_ == 'VBZ' else 'past' if verb.tag_ == 'VBD' else 'other'
        verb_obj_list = [child.text for child in verb.children if ("obj" in child.dep_)]
    # token of interest is object
    elif token.dep_ in ["dobj", "attr"]:
        verb = token.head
        verb_tense = 'present' if verb.tag_ == 'VBZ' else 'past' if verb.tag_ == 'VBD' else 'other'
        verb_subj = verb.head.text

    return verb, verb_tense, verb_obj_list, verb_subj


# for given token, extract (if it's a possessive determiner) what it possesses and if relevant what possesses it
def get_possession_data(token):
    possessed_by, possessed = None, None
    if token.dep_ == "poss":
        possessed = token.head.text
    
    # will assume for now that there is only one possessor...
    for child in token.children:
        if child.dep_ == "poss":
            possessed_by = child.text

    return possessed_by, possessed


# load spacy model
nlp = spacy.load("en_core_web_md")

# load corpus
congressional_df = congress_utils.load_full_df_from_raw(INPUT, remove_procedural_speeches=True)
if DEBUG:
    congressional_df = congressional_df.sample(5000)

# iterate through examples
result_df = []
start = time.time()
print("Looking for instances of God and applying SpaCy analyses")
for idx,row in tqdm(congressional_df.iterrows(), total=len(congressional_df.index)):
    doc = nlp(row['text'])
    # check if token is in doc
    if not any(isinstance_of_interest(token) for token in doc):
        continue

    for token in doc:
        if isinstance_of_interest(token):
            token_type = token.dep_

            # 3 important aspects: verb, possession, adjectives
            verb, verb_tense, verb_obj_list, verb_subj = get_verb_data(token)
            possessed_by, possessed = get_possession_data(token)
            adjectives = [child.text for child in token.children if child.dep_ == "amod"]

            result_df.append({
                'speech_id': idx,
                'token': token.text,
                'token_idx': token.i,
                'token_type': token_type,
                'verb': verb.text if verb else None,
                'verb_tense': verb_tense,
                'verb_obj_list': verb_obj_list,
                'verb_subj': verb_subj,
                'adjectives': adjectives,
                'possessed_by': possessed_by,
                'possessed': possessed,
            })
end = time.time()
print("Time taken to analyze instances: ", end-start)

# dump output
result_df = pd.DataFrame(result_df)
result_df['run_id'] = RUN_ID
result_df.to_csv(OUTPUT_PATH + f"output_{RUN_ID}.csv", index=False)

