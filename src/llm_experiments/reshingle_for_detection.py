import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

import argparse

from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from torch.utils.data import DataLoader, dataloader
import nltk
import torch
from tqdm import tqdm

from data import congress_utils
from src.llm_experiments.postprocess_llm_outputs import load_all_annotations
from src.references.train_biencoder import BiModel
from src.references.retrieve_and_infer import load_bible_data_for_references

import os
import gc

MIN_SENTENCE_LENGTH = 5
BATCH_SIZE = 512


def shingle_cr_sentences_and_match(
    verse_df, 
    congress_df, 
    N_GRAM_SIZE=5,
    MIN_SHARED_NGRAMS=4,
    return_all=False,
    text_col='text',
    verse_text_col='text',
    verse_citation_col='citation',
    congress_idx_col='idx',
    batch_size=1000,
    checkpoint_dir=None,
    checkpoint_frequency=10,
    output_file=None
):
    """
    Function to compare Congress sentences and Bible verses using n-gram shingling.
    
    Args:
        verse_df: Pandas DataFrame containing Bible verses
        congress_df: Pandas DataFrame containing congressional data
        N_GRAM_SIZE: Size of n-grams to use for shingling (default: 5)
        MIN_SHARED_NGRAMS: Minimum number of shared n-grams to consider a match (default: 4)
        return_all: If True, return all results regardless of match count
        text_col: Column name in congress_df containing text to analyze
        verse_text_col: Column name in verse_df containing verse text
        verse_citation_col: Column name in verse_df containing verse citations
        congress_idx_col: Column name in congress_df containing congress indices
        batch_size: Number of congressional sentences to process at once
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        checkpoint_frequency: How often to save checkpoints (in batches)
        output_file: Path to save results incrementally (None = keep all in memory)
        
    Returns:
        DataFrame containing matching results or file path if output_file is provided
    """

    print(f"Comparing Congress sentences with Bible verses using {N_GRAM_SIZE}-gram shingling")
    print(f"Minimum shared n-grams for a match: {MIN_SHARED_NGRAMS}")
    
    # Setup checkpointing if requested
    checkpoint_state = {
        'batch_idx': 0,
        'results': [],
        'processed_count': 0
    }
    
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Load checkpoint if exists
    if checkpoint_dir:
        import pickle
        checkpoint_file = os.path.join(checkpoint_dir, 'shingling_checkpoint.pkl')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    saved_state = pickle.load(f)
                checkpoint_state.update(saved_state)
                print(f"Resuming from checkpoint: processed {checkpoint_state['processed_count']} items")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    # Setup output file if requested
    if output_file:
        # Create or clear the output file
        if checkpoint_state['batch_idx'] == 0:
            cols = ['congress_idx', 'text', 'most_similar_verse', 'shared_ngrams', 'verse_citation']
            pd.DataFrame(columns=cols).to_csv(output_file, index=False)
            print(f"Created output file: {output_file}")
    
    # Prepare verse data as a dictionary for easier access
    verse_data = {
        'texts': verse_df[verse_text_col].tolist(),
        'citations': verse_df[verse_citation_col].tolist()
    }
    num_verses = len(verse_data['texts'])
    print(f"Processing {num_verses} Bible verses")
    
    # Generate n-grams for all verses once
    verse_ngrams = _generate_verse_ngrams(verse_data['texts'], N_GRAM_SIZE)
    print(f"Generated {N_GRAM_SIZE}-grams for all verses")
    
    # Calculate total number of batches
    total_rows = len(congress_df)
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    # Process congress data in batches
    start_batch = checkpoint_state['batch_idx']
    for batch_idx in tqdm(range(start_batch, total_batches)):
        checkpoint_state['batch_idx'] = batch_idx
        
        # Extract batch data
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = congress_df.iloc[start_idx:end_idx]
        
        batch_texts = batch_df[text_col].tolist()
        batch_indices = batch_df[congress_idx_col].tolist()
        
        # Process batch
        batch_results = _process_single_batch_shingling(
            batch_texts, batch_indices, verse_data, verse_ngrams,
            N_GRAM_SIZE, MIN_SHARED_NGRAMS, return_all
        )
        
        # Save results
        checkpoint_state['processed_count'] += len(batch_texts)
        
        if output_file:
            # Append results to file
            if batch_results:
                pd.DataFrame(batch_results).to_csv(output_file, mode='a', header=False, index=False)
        else:
            # Keep results in memory
            checkpoint_state['results'].extend(batch_results)
        
        # Save checkpoint if requested
        if checkpoint_dir and (batch_idx + 1) % checkpoint_frequency == 0:
            import pickle
            with open(os.path.join(checkpoint_dir, 'shingling_checkpoint.pkl'), 'wb') as f:
                pickle.dump(checkpoint_state, f)
            print(f"Saved checkpoint after batch {batch_idx+1}: processed {checkpoint_state['processed_count']} items")
        
        # Force garbage collection
        gc.collect()
    
    # Save final checkpoint
    if checkpoint_dir:
        import pickle
        with open(os.path.join(checkpoint_dir, 'shingling_checkpoint.pkl'), 'wb') as f:
            pickle.dump(checkpoint_state, f)
        print(f"Saved final checkpoint: processed {checkpoint_state['processed_count']} items")
    
    # Return results
    if output_file:
        print(f"Results saved to {output_file}")
        return output_file
    else:
        if not checkpoint_state['results']:
            print("No matches found")
            return pd.DataFrame(columns=['congress_idx', 'text', 'most_similar_verse', 'shared_ngrams', 'verse_citation'])
        return pd.DataFrame(checkpoint_state['results'])


def _generate_ngrams(text, n):
    """
    Generate n-grams from text.
    Returns a set of n-grams.
    """
    # Normalize text by lowercasing and removing extra whitespace
    text = ' '.join(text.lower().split())
    
    # Generate n-grams
    words = text.split()
    ngrams = set()
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    
    return ngrams


def _generate_verse_ngrams(verse_texts, n):
    """
    Generate n-grams for all verses.
    Returns a list of sets of n-grams, one set per verse.
    """
    return [_generate_ngrams(verse, n) for verse in verse_texts]


def _count_shared_ngrams(text_ngrams, verse_ngrams_list):
    """
    Count shared n-grams between a text and all verses.
    Returns a list of counts, one for each verse.
    """
    return [len(text_ngrams.intersection(verse_ngrams)) for verse_ngrams in verse_ngrams_list]


def _process_single_batch_shingling(batch_texts, batch_indices, verse_data, 
                                   verse_ngrams, n, min_shared_ngrams, 
                                   return_all):
    """
    Process a single batch of congress text using n-gram shingling.
    """
    results = []
    
    # Generate n-grams for all texts in batch
    batch_ngrams = [_generate_ngrams(text, n) for text in batch_texts]
    
    # Compare each text with all verses
    for i, text_ngrams in enumerate(batch_ngrams):
        # Skip empty texts or very short texts that can't form enough n-grams
        if len(text_ngrams) == 0:
            continue
        
        # Count shared n-grams with each verse
        shared_counts = _count_shared_ngrams(text_ngrams, verse_ngrams)
        
        # Find verse with most shared n-grams
        max_shared = max(shared_counts) if shared_counts else 0
        max_index = shared_counts.index(max_shared) if shared_counts else -1
        
        # Add to results if meets threshold or return_all is True
        if (max_shared >= min_shared_ngrams or return_all) and max_index >= 0:
            results.append({
                'congress_idx': batch_indices[i],
                'text': batch_texts[i],
                'most_similar_verse': verse_data['texts'][max_index],
                'shared_ngrams': max_shared,
                'verse_citation': verse_data['citations'][max_index],
            })
    
    return results


def main(args):
    if args.use_test_df:
        _, infer_df = load_all_annotations()
    else:
        print("Loading congressional data")
        congressional_df = congress_utils.load_full_df_from_raw(args.input, remove_procedural_speeches=True)
        print(f"Number of speeches in congressional data: {len(congressional_df.index)}")
        #if args.debug:
        #    print(f"Sampling {args.sample} speeches")
        #    congressional_df = congressional_df.sample(args.sample, random_state=42)

        infer_df = create_inference_df(MIN_SENTENCE_LENGTH, congressional_df)
        print(f"Number of sentences in congressional data: {len(infer_df.index)}")

    verse_df, _, _ = load_bible_data_for_references()


    print("Beginning inference")
    #result_df = embed_cr_sentences_match(
    #    full_model,
    #    model,
    #    verse_result_tuples,
    #    congressLoader,
    #    return_all=True,
    #    COSINE_SIM_THRESHOLD=args.cosine_sim_threshold
    #)

    result_df = shingle_cr_sentences_and_match(
        verse_df=verse_df,        # DataFrame containing Bible verses
        congress_df=infer_df,  # DataFrame containing congressional text
        N_GRAM_SIZE=5,                        # Size of n-grams (default: 5)
        MIN_SHARED_NGRAMS=5,                  # Minimum shared n-grams for a match
        text_col='text',          # Column name for congress text
        verse_text_col='text',   # Column name for verse text 
        congress_idx_col='congress_idx',
        batch_size=1000,
        checkpoint_dir='./checkpoints_shingles',
        checkpoint_frequency=10,
        output_file='./results_test_shingles.csv',
    )

    if isinstance(result_df, str):
        print(f"Results saved to {result_df}")
        return
    if result_df.empty:
        print("No matches found")
        return
    
    print(f"Dumping results to {args.out_dir}")
    result_df.to_json(args.out_dir + "ngram_congress_verse_matches.json", index=False)
    print("Done")

    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/corpora/congressional-record/')
    parser.add_argument('--out_dir', type=str, default='/data/laviniad/congress_errata/references/')
    parser.add_argument('--use_test_df', action='store_true')
    args = parser.parse_args()

    main(args)