import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import gc
import time
import json

from rapidfuzz import fuzz, process
from data.bible_utils import comp_bible_helper
from data.congress_utils import induce_party_and_state
from data.data_utils import get_lexical_overlap

import numpy as np
from scipy.spatial.distance import cosine

from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from torch.utils.data import DataLoader, dataloader
import nltk
import torch

tqdm.pandas()

from data import congress_utils
from src.references.train_biencoder import BiModel

TOKEN_OVERLAP_THRESHOLD = 0.2 # this is used to decide whether to compute cosine similarity between embeddings (expensive) or not
MIN_SENTENCE_LENGTH = 5
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# version options are: 'King James Bible','American Standard Version', 'Douay-Rheims Bible',
# 'Darby Bible Translation', 'English Revised Version', 'Webster Bible Translation', 'World English Bible',
# 'Young's Literal Translation', and 'American King James Version'
def load_bible_data_for_references(version='King James Bible'):
    print("Loading Bible data for version: " + version)
    OPTIONS = ['King James Bible','American Standard Version', 'Douay-Rheims Bible', 'Darby Bible Translation', 
               'English Revised Version', 'Webster Bible Translation', 'World English Bible', 
               'Young\'s Literal Translation', 'American King James Version']
    # if version is "ALL", load all versions and return in verse_df
    print("Loading verses...")
    TAG_RE = re.compile(r'<[^>]+>')

    def remove_tags(text):
        return TAG_RE.sub('', text)

    bible_df = comp_bible_helper()
    pop_verses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/most_popular_verses.csv')
    n = 1000 # VERY generous
    pop_citations = list(pop_verses['verse'].iloc[1:n+2])

    for v in OPTIONS:
        bible_df[v] = bible_df[v].apply(remove_tags) # KJV in this df has italics etc
    bible_df['Verse'] = bible_df['Verse'].apply(lambda x: x.lower())
    limited_bible_df = bible_df[bible_df['Verse'].apply(lambda x: x in pop_citations)]
    if version == "ALL":
        # create dict of sets of verses for each version
        print("Creating dict of sets of verses for each version")
        limited_verses = {r['Verse']: {v: r[v] for v in OPTIONS} for idx, r in limited_bible_df.iterrows()}
    else:
        limited_verses = limited_bible_df[version] # verse texts
    print("Now loading verse data")

    # want all versions' texts to be accessible by verse
    # for verse_df, we want to have a column for each version
    if version == "ALL":
        verse_df = [{'text': t["King James Bible"], 'citation': t['citation'], } for idx,t in limited_bible_df.iterrows()]
        for v in OPTIONS:
            verse_df[v] = verse_df['citation'].apply(lambda x: limited_bible_df[limited_bible_df['citation'] == x][v].values[0]) # maps from citation to version
        limited_verse_to_citation = {}
        for v in OPTIONS:
            limited_verse_to_citation.update(dict(zip([v.lower() for v in limited_bible_df[v]], limited_bible_df['Verse'].apply(lambda x: x.lower()).values)))
        limited_citation_to_verse = dict(zip(limited_bible_df['Verse'].apply(lambda x: x.lower()).values, [v.lower() for v in limited_verses])) # allow texts to be KJV -- not used for detection, but rather for storing some text in result dataset
        verse_df = pd.DataFrame(verse_df)
        return verse_df, limited_verse_to_citation, limited_citation_to_verse
    else:
        verse_df = [{'text': t[version], 'citation': t['citation']} for idx,t in limited_bible_df.iterrows()]
        verse_df = pd.DataFrame(verse_df)
        print(f"Number of verses: {len(verse_df.index)}")
        limited_verse_to_citation = dict(zip([v.lower() for v in limited_verses], limited_bible_df['Verse'].apply(lambda x: x.lower()).values))
        limited_citation_to_verse = {v: k for k,v in limited_verse_to_citation.items()}
        return verse_df, limited_verse_to_citation, limited_citation_to_verse


def fast_cosine(ctext, vtext, congress_embedding, verse_embedding, TOKEN_OVERLAP_THRESHOLD=0.2):
    if ctext == vtext: # actually the same... unlikely given formatting etc
        return 0.0

    if len(set(ctext.split()) & set(vtext.split())) < TOKEN_OVERLAP_THRESHOLD * len(set(vtext.split())):
        return 1.0
    
    cos = cosine(congress_embedding, verse_embedding)
    return cos


def main(TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, args):
    # dump args
    with open(args.out_dir + 'args.txt', 'w') as file:
        file.write(str(args))
        print("Wrote args to file")

    KEYWORD_FILTER_THRESHOLD = args.keyword_fraction_threshold
    if args.device == 'cpu':
        DEVICE = 'cpu'
        BATCH_SIZE = 1
    else:
        DEVICE = 'cuda:' + args.device
        BATCH_SIZE = args.batch_size

# load congressional data
    print("Loading congressional data")
    congressional_df = congress_utils.load_full_df_from_raw(args.input, remove_procedural_speeches=True)
    filtered_df = load_and_filter_cr(args, congressional_df, KEYWORD_FILTER_THRESHOLD)

# retrieve the speeches that contain these keywords and create new df with rows containing sentence + verse + sermon idx of sentence
    infer_df = create_inference_df(MIN_SENTENCE_LENGTH, filtered_df)
    print(f"Number of sentences left: {len(infer_df.index)}")

# load verses
    verse_df, _, _ = load_bible_data_for_references()

# load model
    full_model, model = load_embedding_model(args)
    biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# embed bible verses
    verseLoader = create_verse_loader(BATCH_SIZE, verse_df, biTokenizer)

    verse_result_tuples = get_embedded_verses(full_model, model, verseLoader)

# embed congress sentences
    congressLoader = create_congress_loader(BATCH_SIZE, infer_df, biTokenizer)
    result_df = embed_cr_sentences_and_match(DEVICE, full_model, model, verse_result_tuples, congressLoader, TOKEN_OVERLAP_THRESHOLD)


    # 1s and 0s suggest that cosine sim was not computed
    token_overlap_only = len(result_df[result_df['cosine_similarity'] == 1.0])
    token_overlap_only += len(result_df[result_df['cosine_similarity'] == 0.0])
    computed_cosine_sim = len(result_df) - token_overlap_only
    print(f"Computed cosine similarity for " + str(computed_cosine_sim) + " congress-verse pairs")
    print(f"This is " + str(100 * (computed_cosine_sim / len(result_df))) + "% of all congress-verse pairs")

# dump
    if not args.debug:
        if args.sample < 0:
            result_df.to_csv(args.out_dir + 'results_NEW.csv')
            print(f"Dumped result_df to {args.out_dir}results_NEW.csv")
        else:
            result_df.to_csv(args.out_dir + f'sample{args.sample}_results.csv')
            print(f"Dumped result_df to {args.out_dir}sample{args.sample}_results.csv")
    else:
        print("In debug mode; did not dump results")

def create_verse_loader(BATCH_SIZE, verse_df, biTokenizer):
    print("Creating Bible verse data loader")
    verseDataset = Dataset.from_pandas(verse_df)
    verseDataset = verseDataset.map(lambda x: biTokenizer(x["text"], max_length=512, padding="max_length", truncation=True), num_proc=4)
        
    for col in ['input_ids', 'attention_mask']:
        verseDataset = verseDataset.rename_column(col, 'text'+'_'+col)
            
    verseDataset.set_format(type='torch')
# automatically send to device
    verseLoader = torch.utils.data.DataLoader(verseDataset, batch_size=BATCH_SIZE, shuffle=False)
    return verseLoader

def create_congress_loader(BATCH_SIZE, infer_df, biTokenizer):
    print("Creating Congress data loader")
    congressDataset = Dataset.from_pandas(infer_df)
    congressDataset = congressDataset.map(lambda x: biTokenizer(x["text"], max_length=512, padding="max_length", truncation=True), num_proc=16)
        
    for col in ['input_ids', 'attention_mask']:
        congressDataset = congressDataset.rename_column(col, 'text'+'_'+col)
    congressDataset.set_format(type='torch')

    congressLoader = torch.utils.data.DataLoader(congressDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return congressLoader

def create_inference_df(MIN_SENTENCE_LENGTH, filtered_df):
    infer_df = []
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df.index)):
        sentence_list = nltk.sent_tokenize(row['text'])
        for i,s in enumerate(sentence_list):
            if len(s.split()) >= MIN_SENTENCE_LENGTH:
                infer_df.append({
                    'congress_idx': idx,
                    'text': s
                })

    infer_df = pd.DataFrame(infer_df)
    return infer_df


def filter(speech, threshold):
        overlap, _, _, _ = get_lexical_overlap(speech)
        return overlap > threshold


def load_and_filter_cr(args, congressional_df, KEYWORD_FILTER_THRESHOLD, EMBED_ALL=False):

    if args.debug:
        congressional_df = congressional_df.sample(1000)
        print("In debug mode; sampled 1000 documents from congressional record")
    elif args.sample > 0:
        if args.sample < len(congressional_df.index):
            congressional_df = congressional_df.sample(args.sample)
            print(f"Due to sample being passed, sampled down to {args.sample} instances")
        else:
            print("Sample size is larger than CR; using full CR")
    else:
        print("Did not sample, since argument was not passed; using full CR")

    print("Filtering congressional data by keywords")

    if 'embed_all' in args:
        if not args.embed_all:
            congressional_df['religious'] = congressional_df['text'].progress_apply(lambda x: filter(x, KEYWORD_FILTER_THRESHOLD))
            filtered_df = congressional_df[congressional_df['religious']]
        else:
            filtered_df = congressional_df
    else:
        if not EMBED_ALL:
            congressional_df['religious'] = congressional_df['text'].progress_apply(lambda x: filter(x, KEYWORD_FILTER_THRESHOLD))
            filtered_df = congressional_df[congressional_df['religious']]
        else:
            filtered_df = congressional_df

    print(f"Number of religious speeches: {len(filtered_df.index)}, or {100 * (len(filtered_df.index) / len(congressional_df.index))}% of CR")

    return filtered_df


def embed_cr_sentences_and_match(DEVICE, full_model, model, verse_result_tuples, congressLoader, TOKEN_OVERLAP_THRESHOLD=0.2, COSINE_SIM_THRESHOLD=0.8, return_all=False):
    print("Embedding Congress sentences and retrieving most similar verse")
    print('model device: ' + str(model.device))

    result_df = []
    with torch.no_grad():
        for batch in tqdm(congressLoader):
        #start = time.time()
            input_ids = batch["text_input_ids"].to(model.device)
            attention_masks = batch["text_attention_mask"].to(model.device)
            embedding = model(input_ids, attention_mask=attention_masks)[0]
            mean_pooled = full_model.mean_pooling(embedding, attention_masks).to('cpu').detach().numpy()
            mean_pooled = mean_pooled.tolist()
                #t = [e.replace('\n', ' ').replace('\t', ' ') for e in batch['text']]
            batch_data = list(zip(batch['text'], batch['congress_idx'], list(mean_pooled)))
        #end = time.time()
        #print(f"Time taken to embed batch: {end - start}")

        #start = time.time()
            for b in batch_data:
                m = b[2]

                similarities = [1 - fast_cosine(b[0], verse[0], m, verse[2], TOKEN_OVERLAP_THRESHOLD) for verse in verse_result_tuples]
                max_similarity_index = np.argmax(similarities)
                cosine_sim = similarities[max_similarity_index]
                verse_tuple = verse_result_tuples[max_similarity_index]

                if (cosine_sim > COSINE_SIM_THRESHOLD) or return_all:
                    result_df.append({
                        'congress_idx': b[1], # comes from congress_utils.load_full_df_from_raw(args.input) indices
                        'text': b[0],
                        'most_similar_verse': verse_tuple[0],
                        'cosine_similarity': cosine_sim,
                        'verse_citation': verse_tuple[1],
                    })
            del embedding
            del mean_pooled
            del input_ids
            del attention_masks
            torch.cuda.empty_cache()

    if len(result_df) == 0:
        print("No matches found")
        # empty df with requisite columns
        result_df = pd.DataFrame(columns=['congress_idx', 'text', 'most_similar_verse', 'cosine_similarity', 'verse_citation'])
    else:
        result_df = pd.DataFrame(result_df)
    return result_df

def get_embedded_verses(full_model, model, verseLoader):
    print("Embedding Bible verses")
    verse_result_tuples = []
    with torch.no_grad():
        for batch in tqdm(verseLoader):
            input_ids = batch["text_input_ids"].to(model.device)
            attention_masks = batch["text_attention_mask"].to(model.device)
            embedding = model(input_ids, attention_mask=attention_masks)[0]
            mean_pooled = full_model.mean_pooling(embedding, attention_masks).to('cpu')
            for b in range(0, len(batch['text'])):
                verse_result_tuples.append((batch['text'][b], batch['citation'][b], mean_pooled[b]))
    return verse_result_tuples

def load_embedding_model(args):
    print("Loading model...")
    full_model = BiModel(device='cpu')
    if args.model_dir == 'sentence-transformers/all-mpnet-base-v2':
        print("Device: " + str(args.device))
        full_model = full_model.to(args.device)
        model = full_model.model.to(args.device)
    else:
        checkpoint = torch.load(args.model_dir, map_location='cpu')
        full_model.load_state_dict(checkpoint, strict=False) # position ids weren't saved with state, but they're just the absolute ones
        full_model = full_model.to(args.device)
        model = full_model.model.to(args.device)

    model.eval()
    full_model.eval()
    return full_model,model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="/data/laviniad/sermons-ir/modeling/tuned_mpnet/model.pth", type=str)
    parser.add_argument('--input', default="/data/corpora/congressional-record/", type=str)
    parser.add_argument('--out_dir', default="/data/laviniad/sermons-ir/modeling/mpnet_results/", type=str)
    parser.add_argument('--congress_errata_path', default="/data/laviniad/congress_errata/", type=str)
    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--keyword_fraction_threshold', default=0.0005, type=float)
    parser.add_argument('--embed_all', action="store_true")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--sample', default=-1, type=int)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    main(TOKEN_OVERLAP_THRESHOLD, MIN_SENTENCE_LENGTH, args)
