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
from src.references.train_biencoder import BiModel
from src.llm_experiments.postprocess_llm_outputs import load_all_annotations
from src.references.retrieve_and_infer import load_bible_data_for_references
from src.references.retrieve_and_infer import load_embedding_model, get_embedded_verses

DEVICE = "cuda:0"
MIN_SENTENCE_LENGTH = 5
BATCH_SIZE = 512
MODEL = "sentence-transformers/all-mpnet-base-v2"

def load_model():
    print("Loading model")
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    return model

import torch
from tqdm import tqdm
import gc
from collections import namedtuple
import h5py
import os
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def embed_cr_sentences_and_match(
    DEVICE, 
    full_model, 
    model, 
    verse_result_tuples_or_file, 
    congressLoader, 
    COSINE_SIM_THRESHOLD=0.8, 
    return_all=False,
    checkpoint_dir=None,
    checkpoint_frequency=1000,
    max_gpu_batch_size=None,
    verse_batch_size=10000,
    output_file=None
):
    """
    Memory-optimized function to embed Congress sentences and find most similar verses.
    
    Args:
        DEVICE: The device to run computations on
        full_model: The full model containing pooling methods
        model: The embedding model
        verse_result_tuples_or_file: Either a list of tuples containing (verse_text, verse_citation, verse_embedding)
                                    or a path to an H5 file containing verse data
        congressLoader: DataLoader for congressional data
        TOKEN_OVERLAP_THRESHOLD: Threshold for token overlap
        COSINE_SIM_THRESHOLD: Threshold for cosine similarity
        return_all: If True, return all results regardless of similarity score
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        checkpoint_frequency: How often to save checkpoints (in batches)
        max_gpu_batch_size: Maximum batch size to process on GPU at once (rebatching)
        verse_batch_size: Number of verses to process at once for similarity calculation
        output_file: Path to save results incrementally (None = keep all in memory)
        
    Returns:
        DataFrame containing matching results or file path if output_file is provided
    """
    print("Embedding Congress sentences and retrieving most similar verse")
    print(f'Model device: {model.device}')
    
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
        checkpoint_file = os.path.join(checkpoint_dir, 'embedding_checkpoint.pt')
        if os.path.exists(checkpoint_file):
            try:
                saved_state = torch.load(checkpoint_file)
                checkpoint_state.update(saved_state)
                print(f"Resuming from checkpoint: processed {checkpoint_state['processed_count']} items")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Starting fresh.")
    
    # Setup output file if requested
    if output_file:
        # Create or clear the output file
        if checkpoint_state['batch_idx'] == 0:
            cols = ['congress_idx', 'text', 'most_similar_verse', 'cosine_similarity', 'verse_citation']
            pd.DataFrame(columns=cols).to_csv(output_file, index=False)
            print(f"Created output file: {output_file}")
    
    # Load or prepare verse data
    verse_data = _prepare_verse_data(verse_result_tuples_or_file)
    num_verses = verse_data['embeddings'].shape[0]
    print(f"Loaded {num_verses} verse embeddings")
    
    # Ensure model is on the correct device
    model = model.to(DEVICE)
    
    # Process batches
    with torch.no_grad():
        # Skip already processed batches if resuming
        for batch_idx, batch in enumerate(tqdm(congressLoader), checkpoint_state['batch_idx']):
            checkpoint_state['batch_idx'] = batch_idx
            
            # If max_gpu_batch_size is specified, process in sub-batches to save GPU memory
            if max_gpu_batch_size and len(batch["text_input_ids"]) > max_gpu_batch_size:
                sub_batches = _create_sub_batches(batch, max_gpu_batch_size)
                batch_results = []
                
                for sub_batch in sub_batches:
                    sub_batch_results = _process_single_batch(
                        sub_batch, model, full_model, verse_data, 
                        DEVICE, COSINE_SIM_THRESHOLD, 
                        return_all, verse_batch_size
                    )
                    batch_results.extend(sub_batch_results)
            else:
                batch_results = _process_single_batch(
                    batch, model, full_model, verse_data, 
                    DEVICE, COSINE_SIM_THRESHOLD, 
                    return_all, verse_batch_size
                )
            
            # Save results
            checkpoint_state['processed_count'] += len(batch["text_input_ids"])
            
            if output_file:
                # Append results to file
                if batch_results:
                    pd.DataFrame(batch_results).to_csv(output_file, mode='a', header=False, index=False)
            else:
                # Keep results in memory
                checkpoint_state['results'].extend(batch_results)
            
            # Save checkpoint if requested
            if checkpoint_dir and (batch_idx + 1) % checkpoint_frequency == 0:
                torch.save(checkpoint_state, os.path.join(checkpoint_dir, 'embedding_checkpoint.pt'))
                print(f"Saved checkpoint after batch {batch_idx+1}: processed {checkpoint_state['processed_count']} items")
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save final checkpoint
    if checkpoint_dir:
        torch.save(checkpoint_state, os.path.join(checkpoint_dir, 'embedding_checkpoint.pt'))
        print(f"Saved final checkpoint: processed {checkpoint_state['processed_count']} items")
    
    # Return results
    if output_file:
        print(f"Results saved to {output_file}")
        return output_file
    else:
        if not checkpoint_state['results']:
            print("No matches found")
            return pd.DataFrame(columns=['congress_idx', 'text', 'most_similar_verse', 'cosine_similarity', 'verse_citation'])
        return pd.DataFrame(checkpoint_state['results'])


def _prepare_verse_data(verse_result_tuples_or_file):
    """
    Prepare verse data from either tuples or file.
    Returns a dictionary with 'texts', 'citations', and 'embeddings'.
    """
    if isinstance(verse_result_tuples_or_file, str) and os.path.exists(verse_result_tuples_or_file):
        # Load from H5 file
        verse_data = {}
        with h5py.File(verse_result_tuples_or_file, 'r') as f:
            verse_data['embeddings'] = torch.tensor(f['embeddings'][()])
            # Load texts and citations as needed - these can be very memory-intensive
            # Consider using a reference-only approach if these are very large
            verse_data['texts'] = f['texts'][()]
            verse_data['citations'] = f['citations'][()]
        return verse_data
    else:
        # Process from tuples
        verse_data = {
            'texts': [verse[0] for verse in verse_result_tuples_or_file],
            'citations': [verse[1] for verse in verse_result_tuples_or_file],
            'embeddings': torch.tensor(np.array([verse[2] for verse in verse_result_tuples_or_file]))
        }
        return verse_data


def _create_sub_batches(batch, max_size):
    """
    Split a large batch into smaller sub-batches to save GPU memory.
    """
    total_size = len(batch["text_input_ids"])
    sub_batches = []
    
    for i in range(0, total_size, max_size):
        end = min(i + max_size, total_size)
        sub_batch = {
            "text_input_ids": batch["text_input_ids"][i:end],
            "text_attention_mask": batch["text_attention_mask"][i:end],
            "text": batch["text"][i:end],
            "congress_idx": batch["congress_idx"][i:end]
        }
        sub_batches.append(sub_batch)
    
    return sub_batches


def _process_single_batch(batch, model, full_model, verse_data, DEVICE, 
                          COSINE_SIM_THRESHOLD, 
                          return_all, verse_batch_size):
    """
    Process a single batch of congress text.
    """
    # Move batch data to model device
    input_ids = batch["text_input_ids"].to(DEVICE)
    attention_masks = batch["text_attention_mask"].to(DEVICE)
    
    # Get embeddings
    embedding = model(input_ids, attention_mask=attention_masks)[0]
    
    # Mean pooling
    batch_embeddings = full_model.mean_pooling(embedding, attention_masks)
    
    # Get batch data
    batch_texts = batch['text']
    batch_indices = batch['congress_idx']
    
    # Process batch
    results = []
    
    # Without token overlap, we can process the entire batch at once against verse batches
    batch_embeddings_cpu = batch_embeddings.cpu().numpy()
        
    # Find best matches by processing verse embeddings in batches
    max_similarities = np.full(len(batch_embeddings_cpu), -1.0)
    max_indices = np.full(len(batch_embeddings_cpu), -1)
        
    for verse_start in range(0, len(verse_data['embeddings']), verse_batch_size):
        verse_end = min(verse_start + verse_batch_size, len(verse_data['embeddings']))
        verse_batch_embs = verse_data['embeddings'][verse_start:verse_end].cpu().numpy()
            
        # Calculate batch similarity using sklearn for memory efficiency
        sim_matrix = sklearn_cosine_similarity(batch_embeddings_cpu, verse_batch_embs)
            
        # Find max similarities in this verse batch
        batch_max_indices = np.argmax(sim_matrix, axis=1)
        batch_max_similarities = np.max(sim_matrix, axis=1)
            
        # Update overall max if better match found
        update_mask = batch_max_similarities > max_similarities
        max_similarities[update_mask] = batch_max_similarities[update_mask]
        max_indices[update_mask] = verse_start + batch_max_indices[update_mask]
        
    # Create results for matches
    for i, (similarity, verse_idx) in enumerate(zip(max_similarities, max_indices)):
        if (similarity > COSINE_SIM_THRESHOLD) or return_all:
            results.append({
                'congress_idx': batch_indices[i],
                'text': batch_texts[i],
                'most_similar_verse': verse_data['texts'][verse_idx],
                'cosine_similarity': float(similarity),
                'verse_citation': verse_data['citations'][verse_idx],
            })
    
    # Clean up memory
    del embedding, batch_embeddings, input_ids, attention_masks
    
    return results


def save_verse_data_to_h5(verse_result_tuples, output_file):
    """
    Save verse data to H5 file for more efficient loading.
    
    Args:
        verse_result_tuples: List of tuples containing (verse_text, verse_citation, verse_embedding)
        output_file: Path to save H5 file
    """
    texts = [verse[0] for verse in verse_result_tuples]
    citations = [verse[1] for verse in verse_result_tuples]
    embeddings = np.array([verse[2] for verse in verse_result_tuples])
    
    # Save to H5 file
    with h5py.File(output_file, 'w') as f:
        string_dt = h5py.special_dtype(vlen=str)
        
        # Create datasets
        f.create_dataset('embeddings', data=embeddings)
        
        # Create string datasets
        texts_dataset = f.create_dataset('texts', (len(texts),), dtype=string_dt)
        citations_dataset = f.create_dataset('citations', (len(citations),), dtype=string_dt)
        
        # Fill string datasets
        for i, (text, citation) in enumerate(zip(texts, citations)):
            texts_dataset[i] = text
            citations_dataset[i] = citation
    
    print(f"Saved verse data to {output_file}")
    return output_file


def cosine_similarity(A, B):
    # Normalize the vectors for cosine similarity
    A_normalized = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)
    
    return np.dot(A_normalized, B_normalized.T)


def main(args):
    print("Loading congressional data")
    if args.use_test_df:
        _, infer_df = load_all_annotations()
        infer_df = infer_df[['congress_idx', 'text']]
        
        print(f"Number of sentences in congressional data: {len(infer_df.index)}")
    else:
        congressional_df = congress_utils.load_full_df_from_raw(args.input, remove_procedural_speeches=True)
        print(f"Number of speeches in congressional data: {len(congressional_df.index)}")
        if args.debug:
            print(f"Sampling {args.sample} speeches")
            congressional_df = congressional_df.sample(args.sample, random_state=42)

        infer_df = create_inference_df(MIN_SENTENCE_LENGTH, congressional_df)
        print(f"Number of sentences in congressional data: {len(infer_df.index)}")

    verse_df, _, _ = load_bible_data_for_references()

    # load model
    print("Loading model...")
    full_model, model = load_embedding_model('sentence-transformers/all-mpnet-base-v2', DEVICE)
    biTokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    print("Loading verses...")
    verseLoader = create_verse_loader(BATCH_SIZE, verse_df, biTokenizer)
    verse_result_tuples = get_embedded_verses(full_model, model, verseLoader)
    # Save verse data to H5 file format for more efficient loading
    verse_data_file = save_verse_data_to_h5(verse_result_tuples, "verse_data.h5")

    print("Creating Congress loader")
    congressLoader = create_congress_loader(BATCH_SIZE, infer_df, biTokenizer)

    for i in range(len(congressLoader.dataset)):
        item = congressLoader.dataset[i]
        if item is None:
            print(f"Item at index {i} is None")
        if not isinstance(item, dict) or any(v is None for v in item.values()):
            print(f"Corrupt item at index {i}: {item}")
    

    print("Beginning inference")
    #result_df = embed_cr_sentences_match(
    #    full_model,
    #    model,
    #    verse_result_tuples,
    #    congressLoader,
    #    return_all=True,
    #    COSINE_SIM_THRESHOLD=args.cosine_sim_threshold
    #)

    result_df = embed_cr_sentences_and_match(
        DEVICE, full_model, model, verse_data_file, 
        congressLoader,
        max_gpu_batch_size=1000,    # Process at most 1k samples on GPU at once
        verse_batch_size=10000,    # Process verse similarity in chunks of 10000
        checkpoint_dir="./checkpoints",  # Enable checkpointing
        checkpoint_frequency=50,  # Save progress every 100 batches
        output_file="./results_test_emb.csv"  # Save results incrementally
    )

    if not args.debug:
        print(f"Dumping results to {args.out_dir}")
        result_df.to_json(args.out_dir + "congress_verse_matches.json", index=False)
        print("Done")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/data/corpora/congressional-record/')
    parser.add_argument('--out_dir', type=str, default='/data/laviniad/congress_errata/references/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--cosine_sim_threshold', type=float, default=0.75)
    parser.add_argument('--use_test_df', action='store_true')
    args = parser.parse_args()

    main(args)