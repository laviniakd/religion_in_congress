## build POS model from list of texts
import nltk
import argparse
import os
from data.load_reference_targets_from_original_df import clean
from itertools import chain
from multiprocessing import Pool, cpu_count
from data import coca_utils, presidential_utils
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
import pandas as pd
import random
from collections import defaultdict
import json
import spacy
import pickle as pkl
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', default="/data/laviniad/sermons-ir/sermoncentral/sermons_clean_text.csv", type=str)
parser.add_argument('--outdir', default="/data/laviniad/sermons-ir/POS_ngrams/", type=str)
parser.add_argument('--name', default="sermons", type=str)
parser.add_argument('--n', default=2, type=int)
parser.add_argument('--nproc', default=8, type=int)
parser.add_argument('--tags_path', default='', type=str)
parser.add_argument('--ngram_path', default='', type=str)
parser.add_argument('--chunksize', default=1000, type=int)
parser.add_argument('--dump_ngrams', action='store_true')
parser.add_argument('--max_length', default=1000000, type=str)
args = parser.parse_args()

MAX_LEN = args.max_length
DUMP_NGRAMS = args.dump_ngrams

# data
print("Loading data...")
if args.tags_path == '':
    if args.name == 'sermons':
        sermon_df = pd.read_csv(args.input, header=None)
    
        sermon_df.columns = ['index', 'link', 'denomination',
                                 'author', 'churchName', 'churchAddress',
                                 'unknown1', 'unknown2', 'unknown3',
                                 'date', 'title', 'versesList',
                                 'topicsList', 'ratingsNumber',
                                 'rating', 'text']
    
        sermon_df.set_index('index', inplace=True)
        texts = list(sermon_df['text'])
    elif args.name.lower() == 'coca':
        result_dict = coca_utils.load_COCA_from_raw(args.input)
        coca_df = coca_utils.output_comprehensive_df(result_dict)
        texts = list(coca_df['text'])
    else:
        result_dict = presidential_utils.load_presidential_from_raw(args.input)
        pres_df = presidential_utils.output_comprehensive_df(result_dict)
        texts = list(pres_df['text'])

    text_to_idx = {t: i for i,t in enumerate(texts)}
    idx_to_text = {i: t for i,t in enumerate(texts)}

if args.tags_path == '':
    with open(args.outdir + '/' + args.name + '_text_to_idx.json', 'w') as f:
        json.dump(text_to_idx, f)
    with open(args.outdir + '/' + args.name + '_idx_to_text.json', 'w') as f:
        json.dump(idx_to_text, f)


def tag_text(text_chunk):
    tag_chunk = []
    for sermon in tqdm(nltk.sent_tokenize(str(text_chunk))):
        tagged_text = nltk.pos_tag(nltk.word_tokenize(sermon))
        tagged_text = [e[1] for e in tagged_text]
        tag_chunk.append(tagged_text)
    return tag_chunk
    

if args.tags_path == '' and args.ngram_path == '':
    
    print("Now processing...")
    idx_to_tags = []
    count = 0

    # parallelize!
    chunks = [texts[i:i + args.chunksize] for i in range(0, len(texts), args.chunksize)]
    pool = Pool()
    pos_chunks = pool.map(tag_text, chunks)
    tags = [item for sublist in pos_chunks for item in sublist]
    
    print("Dumping tags...")
    with open(args.outdir + '/' + args.name + '_tag_lists.pkl', 'wb') as f:
        pkl.dump(tags, f)
        print(f"Dumped POS tags to {f.name}")
elif args.ngram_path == '':
    with open(args.tags_path, 'rb') as f:
        tags = pkl.load(f)
    
# create ngrams
print("Creating ngrams...")
def make_ngrams(chunk):
    return [list(nltk.ngrams(pos_of_sermon, args.n)) for pos_of_sermon in tqdm(chunk)]

if args.ngram_path == '':
    #tags = [[e[1] for e in d] for d in idx_to_tags]
    pos_tag_chunks = [tags[i:i +  args.chunksize] for i in range(0, len(tags), args.chunksize)]
    pool = Pool()
    ngram_chunks = pool.map(make_ngrams, pos_tag_chunks)
    ngrams = [item for sublist in ngram_chunks for item in sublist]
else:
    with open(args.ngram_path, 'rb') as f:
        ngrams = pkl.load(f)
# dump if want to
if DUMP_NGRAMS:
    with open(f'{args.outdir}raw_ngrams/{args.name}_{args.n}.pkl', 'wb') as f:
        pkl.dump(ngrams, f)
        print(f"Dumped ngrams to {f.name}")

# rewrite!
ngrams = [tuple(e) for e in ngrams] # needs to be hashable
#print(f"At line ~110, ngrams sample: {ngrams[:5]}")
#ngram_counts = nltk.FreqDist(ngrams)

# probabilities
print("Computing probabilities...")
def build_ngram_model(ngrams):
    ngram_counts = defaultdict(int)
    n_1gram_counts = defaultdict(int)
    ngram_probabilities = {}

    #print("building ngram model")
    for i,sermon in tqdm(enumerate(ngrams)):
        for ngram in list(sermon):
            ngram = list(ngram)
            if args.name != 'sermons':
                ngram_counts[' '.join(ngram)] += 1 # transform into strings, not ordered lists
                n_1gram = ' '.join(ngram[:-1])
                n_1gram_counts[n_1gram] += 1
            else:
                for n in ngram: 
                    n = list(n)
                    ngram_counts[' '.join(n)] += 1
                    n_1gram = ' '.join(n[:-1])
                    n_1gram_counts[n_1gram] += 1

    for ngram, count in tqdm(ngram_counts.items()):
        n_1gram = ' '.join(ngram.split()[:-1])
        probability = count / n_1gram_counts[n_1gram] # normalize
        ngram_probabilities[ngram] = probability

    return ngram_probabilities
    
ngram_probabilities = build_ngram_model(ngrams)
print(f"Probability sample: {random.sample(list(ngram_probabilities.items()), 5)}")
    
print("Dumping probabilities...")
with open(args.outdir + args.name + f'_{args.n}.json', 'w') as f:
    json.dump(ngram_probabilities, f)
