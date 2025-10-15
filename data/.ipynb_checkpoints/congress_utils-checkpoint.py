## data util file for congressional record in /data/corpora/congressional-record
import os
from tqdm import tqdm
import csv
import multiprocessing
import warnings
import little_mallet_wrapper as lmw
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import re

tqdm.pandas()

YEAR_RANGE = [str(yr) for yr in range(1990, 2018)]

# produces token --> weight dict for logistic classifier
def load_procedural_weights(path='/home/laviniad/projects/religion_in_congress/data/procedural/model.nontest.tsv'):
    weights = pd.read_csv(path, delimiter='\t')
    weights = weights.to_dict(orient='index')
    weights = {v['Unnamed: 0']: v['yes'] for k,v in weights.items()}
    
    return weights

PROC_WEIGHTS = load_procedural_weights()

def multi_filter(x):
    return filter_congress_with_procedural_classifier(x, PROC_WEIGHTS)


# no cleaning for now
def prep_congress(txt_file):
    return txt_file


def load_bioguide_id_to_dw_nominate(path='/data/laviniad/congress_errata/HSall_members.csv', first_year=1980):
    data = pd.read_csv(path)
    data = data[data['died'] >= first_year - 1] # don't need EVERY congressperson -- if they died before the given year, probably safe
    return {row['bioguide_id']: [row['nominate_dim1'], row['nominate_dim2']] for idx, row in data.iterrows()}


def induce_party_and_state(congressional_df):
    bio_dict = load_bios()
    def get_party(x):
        if x in bio_dict.keys():
            return bio_dict[x]['party']
        else:
            return 'unknown'
    
    
    def get_state(x):
        if x in bio_dict.keys():
            if bio_dict[x]['jobPositions']:
                congress_pos = bio_dict[x]['jobPositions'][0]['congressAffiliation']
                if 'represents' in congress_pos.keys():
                    return congress_pos['represents']['regionCode']
            return 'unknown'
        else:
            return 'unknown'
        
    congressional_df['party'] = congressional_df['bio_id'].apply(get_party)
    congressional_df['state'] = congressional_df['bio_id'].apply(get_state)
    return congressional_df


def induce_topic(congressional_df, topic_distributions_path='/data/laviniad/sermons-ir/topic_models/congress/', topic_distributions_suffix='mallet.topic_distributions.60'):
    topic_labels = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/labeled_topics.csv', header=None)
    topic_labels = {row[0]: row[3] for idx,row in topic_labels.iterrows()} # idx to name
    sample = pd.read_csv('/data/laviniad/congress_errata/small_congress.csv')
    tds = lmw.load_topic_distributions(topic_distributions_path + topic_distributions_suffix) 
    
    def get_main_topic(index):
        if index in sample.index:
            dist = tds[index]
            main_topic = topic_labels[np.argmax(dist)]
            return main_topic
        else:
            return 'nonlabeled'
    
    congressional_df['main_topic'] = congressional_df.index.map(get_main_topic)
    return congressional_df


def load_full_df_from_raw(root, truncate_length=-1, tokenizer=None, debug=False, remove_procedural_speeches=False, nonprocedural_indices_path='/data/laviniad/congress_errata/nonprocedural_indices.json'):
    full_df = []
    
    for subdir, dirs, files in os.walk(root):
        for idx, file in enumerate(files):
            print(f"On file {idx}")
            if file.endswith('.jsonlist'):
                file_path = os.path.join(subdir, file)
                with open(file_path) as f:
                    lines = f.readlines()
                if debug:
                    lines = lines[:5]

                for l in tqdm(lines):
                    try:
                        obj = json.loads(l)
                        results_dict = {
                                'bio_id': obj['bioguide'],
                                'congress_num': obj['congress'],
                                'year': obj['year'],
                                'date': obj['date'],
                                'chamber': obj['chamber'],
                                'speaker': obj['speaker'], # nondistinguishing
                                'text': obj['text']
                        }
                        if truncate_length > 0 and tokenizer is not None:
                            truncated_and_tokenized_text = tokenizer.tokenize(results_dict['text'])[:truncate_length]              
                            token_ids = tokenizer.convert_tokens_to_ids(truncated_and_tokenized_text)
                            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                            results_dict['trunc_and_tokenized_text'] = decoded_text
                                
                        full_df.append(results_dict)
                    except json.decoder.JSONDecodeError:
                        print("couldn't decode from file " + file_path)

    full_df = pd.DataFrame(full_df)
    
    if remove_procedural_speeches:
        if nonprocedural_indices_path is not None:
            with open(nonprocedural_indices_path) as f:
                indices = json.load(f)
            full_df = full_df[indices]
        else:
            tqdm.pandas()
            orig_length = len(full_df.index)
            print("Now filtering with procedural classifier")
            
            # possibly encountering bug?
            with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 10)) as pool:
                results = pool.map(multi_filter, full_df['text'])

            full_df = full_df[results] # apply boolean array
            print(f"Removed {orig_length - len(full_df.index)} procedural speeches, or {(orig_length - len(full_df.index)) / orig_length}%")
            with open('/data/laviniad/congress_errata/nonprocedural_indices.json', 'w') as f:
                json.dump(list([bool(r) for r in results]), f)
                print(f"Dumped nonprocedural indices to {f.name}")
        
    return full_df


def load_congress_from_raw(root='/data/corpora/congressional-record/', debug=True):
    print('Note that this function outputs text that is already joined')
    file_count = sum(len(files) for _, _, files in os.walk(root))
    year_dict = {y: '' for y in YEAR_RANGE}
    for subdir, dirs, files in tqdm(os.walk(root), total=file_count):
        for file in files: # should be just 1 for each genre directory
            if file.endswith('.jsonlist'):
                file_path = os.path.join(subdir, file)
                with open(file_path) as f:
                    lines = f.readlines()

                for l in lines:
                    try:
                        obj = json.loads(l)
                        year = obj['year']
                        speaker = obj['speaker']
                        if str(year) in YEAR_RANGE:
                            year_dict[str(year)] += prep_congress(obj['text']) + '\n'
                    except json.decoder.JSONDecodeError:
                        print("couldn't decode from file " + file_path)

    return year_dict


def load_bios(file='/data/laviniad/congress_bioguides.jsonlist'):
    bio_dict = {}
    
    with open(file) as f:
        lines = f.readlines()

        for l in lines:
            obj = json.loads(l)
            bio_id = obj['usCongressBioId']
            del obj['usCongressBioId']
            bio_dict[bio_id] = obj

    return bio_dict


# turns year -> genre -> text dict into one dataframe
def output_comprehensive_df(congress_dicts, output_path=None):
    full_df = []

    for year in YEAR_RANGE:
        full_df.append({'year': year, 'text': congress_dicts[year]})

    full_df = pd.DataFrame(full_df)
    if output_path is not None:
        full_df.to_csv(output_path, index=False)
    return full_df


def sigmoid(x):
    return 1/(1 + np.exp(-x)) 


# returns True if NOT procedural
def filter_congress_with_procedural_classifier(speech_text, weights):
    warnings.filterwarnings('ignore')
    bias = weights['__BIAS__']
    weight_list = list(weights.items())
    vocab, weight_vals = [e[0] for e in weight_list], [e[1] for e in weight_list]
    tokenized_speech = [s.lower() for s in nltk.wordpunct_tokenize(speech_text)]
    unigrams = set(tokenized_speech)
    bigrams = nltk.ngrams(tokenized_speech, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
    bigrams = set(['_'.join(b) for b in bigrams])
    trigrams = nltk.ngrams(tokenized_speech, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
    trigrams = set(['_'.join(b) for b in trigrams])
    all_grams = unigrams.union(bigrams).union(trigrams)
    
    score = bias
    vectorized = [1 if v in all_grams else 0 for v in vocab]
    score += np.dot(vectorized, weight_vals)
        
    y_pred = sigmoid(score)
    pred = y_pred < 0.5
    return pred
