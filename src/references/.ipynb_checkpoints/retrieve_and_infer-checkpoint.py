import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import gc

from rapidfuzz import fuzz, process
from data.bible_utils import comp_bible_helper
from data.congress_utils import induce_party_and_state
from data.data_utils import get_lexical_overlap

import numpy as np
from scipy.spatial.distance import cosine

from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader, dataloader
import nltk
import torch

tqdm.pandas()

from data import congress_utils
from src.references.train_biencoder import BiModel

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/modeling/mpnet_results/", type=str)
parser.add_argument('--congress_errata_path', default="/data/laviniad/congress_errata/", type=str)
parser.add_argument('--device', default="0", type=str)
parser.add_argument('--keyword_fraction_threshold', default=0.0005, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--sample', default=-1, type=int)
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

KEYWORD_FILTER_THRESHOLD = args.keyword_fraction_threshold
DEVICE = torch.device('cuda:' + args.device)
BATCH_SIZE = args.batch_size

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
    overlap, _ = get_lexical_overlap(speech)
    return overlap > threshold

congressional_df['religious'] = congressional_df['text'].progress_apply(lambda x: filter(x, KEYWORD_FILTER_THRESHOLD))
filtered_df = congressional_df[congressional_df['religious']]
print(f"Number of religious speeches: {len(filtered_df.index)}, or {100 * (len(filtered_df.index) / len(congressional_df.index))}% of CR")

infer_df = []
for idx, row in tqdm(filtered_df.iterrows()):
    sentence_list = nltk.sent_tokenize(row['text'])
    for i,s in enumerate(sentence_list):
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

bible_df = comp_bible_helper()
pop_verses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv')
n = 500 # VERY generous
pop_citations = list(pop_verses['verse'].iloc[1:n+1])
bible_df['King James Bible'] = bible_df['King James Bible'].apply(remove_tags) # KJV in this df has italics etc
bible_df['Verse'] = bible_df['Verse'].apply(lambda x: x.lower())
limited_bible_df = bible_df[bible_df['Verse'].apply(lambda x: x in pop_citations)]
limited_verses = limited_bible_df['King James Bible']
verse_df = [{'text': t['King James Bible'], 'citation': t['citation']} for idx,t in limited_bible_df.iterrows()]
verse_df = pd.DataFrame(verse_df)
print(f"Number of verses: {len(verse_df.index)}")
limited_verse_to_citation = dict(zip(limited_verses, limited_bible_df['Verse']))
limited_citation_to_verse = {v.lower(): k for k,v in limited_verse_to_citation.items()}

# load model
print("Loading model...")
full_model = BiModel(device='cpu')
checkpoint = torch.load(args.model_dir, map_location='cpu')
full_model.load_state_dict(checkpoint, strict=False) # position ids weren't saved with state, but they're just the absolute ones

# having memory issues
print("Device: " + str(DEVICE))
full_model = full_model.to(DEVICE)
model = full_model.model.to(DEVICE)
#model = model.to(DEVICE)

model.eval()
full_model.eval()
biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# embed congress sentences
print("Creating Congress data loader")
congressDataset = Dataset.from_pandas(infer_df)
congressDataset = congressDataset.map(lambda x: biTokenizer(x["text"], max_length=512, padding="max_length", truncation=True))
        
for col in ['input_ids', 'attention_mask']:
    congressDataset = congressDataset.rename_column(col, 'text'+'_'+col)
            
congressDataset.set_format(type='torch')
congressLoader = torch.utils.data.DataLoader(congressDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("Embedding Congress sentences")
congress_result_tuples = []

print('model device: ' + str(model.device))
#print('full_model device: ' + str(full_model.device))
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with torch.no_grad():
        for batch in tqdm(congressLoader):
            input_ids = batch["text_input_ids"].to(DEVICE)
            attention_masks = batch["text_attention_mask"].to(DEVICE)
            embedding = model(input_ids, attention_mask=attention_masks)[0]
            mean_pooled = full_model.mean_pooling(embedding, attention_masks).to('cpu')
            batch_data = list(zip(batch['text'], batch['congress_idx'], list(mean_pooled)))
            congress_result_tuples += batch_data
        #for b in range(0, len(batch['text'])):
        #    congress_result_tuples.append((batch['text'][b], batch['sermon_idx'][b], mean_pooled[b]))
print(prof)
    
# embed bible verses
print("Creating Bible verse data loader")
verseDataset = Dataset.from_pandas(verse_df)
verseDataset = verseDataset.map(lambda x: biTokenizer(x["text"], max_length=512, padding="max_length", truncation=True))
        
for col in ['input_ids', 'attention_mask']:
    verseDataset = verseDataset.rename_column(col, 'text'+'_'+col)
            
verseDataset.set_format(type='torch')
# automatically send to device
verseLoader = torch.utils.data.DataLoader(verseDataset, batch_size=BATCH_SIZE, shuffle=False)

print("Embedding Bible verses")
verse_result_tuples = []
with torch.no_grad():
    for batch in tqdm(verseLoader):
        input_ids = batch["text_input_ids"].to(DEVICE)
        attention_masks = batch["text_attention_mask"].to(DEVICE)
        embedding = model(input_ids, attention_mask=attention_masks)[0]
        mean_pooled = full_model.mean_pooling(embedding, attention_masks).to('cpu')
        for b in range(0, len(batch['text'])):
            verse_result_tuples.append((batch['text'][b], batch['citation'][b], mean_pooled[b]))

# create new df of references given the above pairs
result_df = []
print("Finding most similar Bible verses")
for congress_tuple in tqdm(congress_result_tuples):
    embedding = congress_tuple[2]
    
    similarities = [1 - cosine(embedding, verse[2]) for verse in verse_result_tuples]
    max_similarity_index = np.argmax(similarities)
    cosine_sim = similarities[max_similarity_index]
    verse_tuple = verse_result_tuples[max_similarity_index]
    result_df.append({
        'congress_idx': congress_tuple[1], # comes from congress_utils.load_full_df_from_raw(args.input) indices
        'text': congress_tuple[0],
        'most_similar_verse': verse_tuple[0],
        'cosine_similarity': cosine_sim,
        'verse_citation': verse_tuple[1],
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
