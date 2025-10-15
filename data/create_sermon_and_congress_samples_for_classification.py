import pandas as pd
from tqdm import tqdm
from data import congress_utils, data_utils
import nltk

tqdm.pandas()

N = 25000
OUT_DIR = '/data/laviniad/religious_speech_classifier_data/'
CONGRESS_PATH = '/data/corpora/congressional-record/'

sermon_data = data_utils.load_sermons()
congress_data = congress_utils.load_full_df_from_raw(CONGRESS_PATH, remove_procedural_speeches=True)

sermon_eval_data = sermon_data.sample(n=N, random_state=42)
sermon_eval_data['sentences'] = sermon_eval_data['text'].progress_apply(nltk.sent_tokenize)
sermon_eval_data = sermon_eval_data.explode('sentences').reset_index(drop=True)
sermon_eval_data['label'] = 1
congress_eval_data = congress_data.sample(n=N, random_state=42)
congress_eval_data['sentences'] = congress_eval_data['text'].progress_apply(nltk.sent_tokenize)
congress_eval_data['congress_idx'] = congress_eval_data.index
congress_eval_data = congress_eval_data.explode('sentences').reset_index(drop=True)
congress_eval_data['label'] = 0

sermon_eval_data.drop('text', axis=1, inplace=True)
congress_eval_data.drop('text', axis=1, inplace=True)

sermon_eval_data.to_csv(OUT_DIR + 'sermon_sample.csv')
congress_eval_data.to_csv(OUT_DIR + 'congress_sample.csv')

print("Saved samples to " + OUT_DIR)

