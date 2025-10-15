import pandas as pd
import numpy as np
from collections import Counter
import json
from tqdm import tqdm
import time
import tiktoken
import argparse
import os
import sys
import re

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


import data.bible_utils as bible_utils

tqdm.pandas()

def process_br_detect_response(text, output_format=["YES", "NO"]):
    if not text or text.strip() == "":
        return None
    if not output_format:
        output_format = ["YES", "NO"]

    t = text.strip()
    if t == "YES":
        return 1
    elif t == "NO":
        return 0
    
    try:
        #print(text.strip())
        assert (t in output_format), f"Unexpected response: {t}"
        if t == output_format[0]:
            return 1
        elif t == output_format[1]:
            return 0
        else:
            raise ValueError(f"Unexpected response: {t}")
    except AssertionError as e:
        if output_format[0] in t:
            return 1
        elif output_format[1] in t:
            return 0
        
        return None


def process_br_detect_id_response(text, output_format=["YES", "NO"], json_formatted=False):
    prefaces = ["OUTPUT:", "SOLUTION:", "ANSWER:", "TASK 1:"]
    if not text:
        return None, None
    
    if json_formatted:
        try:
            data = json.loads(text)
            if 'is_verse' not in data or 'verse_citation' not in data:
                return None, None
            return int(data['is_verse'] == "YES"), data['verse_citation']
        except json.JSONDecodeError as e:
            return None, None
    
    try:
        responses = text.strip().split("\n")
        for p in prefaces:
            if responses[0].strip().startswith(p):
                responses[0] = responses[0].strip()[len(p):]

        responses = [r.strip() for r in responses if "TASK 2:" not in r and r.strip() != ""]  
        
        if len(responses) == 0:
            return None, None

        assert(len(responses) > 1), f"Unexpected response format: {text.strip()}"
        is_br = process_br_detect_response(responses[0], output_format)
        if is_br is None and output_format is not None:
            if output_format[0] in text:
                is_br = 1
            elif output_format[1] in text:
                is_br = 0

        br_id = '\n'.join(responses[1:]).strip()
        return is_br, br_id
    except AssertionError as e:
        return None, None


def process_rc_response(text):
    if not text:
        return None
    
    try:
        assert (text.strip() == "RELIGIOUS" or text.strip() == "SECULAR"), f"Unexpected response: {text.strip()}"
        return int(text.strip() == "RELIGIOUS")
    except AssertionError as e:
        if "RELIGIOUS" in text:
            return 1
        elif "SECULAR" in text:
            return 0
        else:
            return None
        

def find_keywords(text):
    if not text:
        return None
    
    try:
        print("text: ", text)
        list_regex = r'\[(.*?)\]'
        match = re.search(list_regex, text)
        print("match: ", match)

        no_bracket_list = r'([a-zA-Z0-9_]+,?\s)+'
        nb_match = re.search(no_bracket_list, text)
        print("nb_match: ", nb_match)
        if match:
            list_str = match.group(0)
            list_str = list_str.replace("'", '"')  # replace single quotes with double quotes
            print("list_str: ", list_str)
            keywords = json.loads(list_str) # as list
            print("keywords: ", keywords)
            assert(isinstance(keywords, list)), f"Expected keywords to be a list: {keywords}"
        elif nb_match:
            list_str = nb_match.group(0)
            list_str = list_str.replace("'", '"')
            keywords = json.loads('[' + list_str + ']')
            assert(isinstance(keywords, list)), f"Expected keywords to be a list: {keywords}"
        else:
            keywords = None
    except json.JSONDecodeError as e:
        keywords = []
    
    return keywords


def process_rc_kw_response(text):
    prefaces = ["OUTPUT:", "SOLUTION:", "ANSWER:", "TASK 1:", "TASK 2:"]
    if not text:
        return None
    
    for p in prefaces:
        text = text.replace(p, "\n")
    
    try:
        responses = text.strip().split("\n")
        #print("Response before: ", responses)
        responses = [r.strip() for r in responses if (r.strip() != "")]
        #print("Response after: ", responses)
        #assert(len(responses) == 2), f"Unexpected response format: {text.strip()}"
        is_religious = process_rc_response(responses[0])
        if is_religious is None:
            if "RELIGIOUS" in text:
                is_religious = 1
            elif "SECULAR" in text:
                is_religious = 0
            else:
                return None, None
        if is_religious and len(responses) > 1:
            keywords = find_keywords(' '.join(responses[1:]))
            
        elif is_religious:
            # after label
            if 'RELIGIOUS' in text:
                keywords = find_keywords(text.split('RELIGIOUS')[1])
            elif 'SECULAR' in text:
                keywords = find_keywords(text.split('SECULAR')[1])
            else:
                keywords = []
        else:
            keywords = None
            
        return is_religious, keywords
    except AssertionError as e:
        return None, None
    

def load_data_together(directory, log_path):
    results_dict = {}

    for file in os.listdir(directory):
        if file.endswith(".json"):
            df = pd.read_json(os.path.join(directory, file))
            run_id = file.split("run_outputs_")[1].split(".json")[0]
            log_file = os.path.join(log_path, f"llm_log_{run_id}.json")
            with open(log_file, "r") as f:
                log_data = json.load(f)
            results_dict[run_id] = {"df": df, "log": log_data}

    return results_dict


def verse_correct(verse, correct_verse_citation):
    if verse is None or correct_verse_citation is None:
        return False
    verse = verse.lower()
    correct_verse_citation = correct_verse_citation.lower()
    if verse == correct_verse_citation:
        return True
    elif verse in correct_verse_citation:
        return True
    elif correct_verse_citation in verse:
        return True
    elif verse.replace(" ", "") == correct_verse_citation.replace(" ", ""):
        return True

    return False
    

def load_br_labels(label_dir) -> dict:
    if isinstance(label_dir, str): # can be str for directory or list of directories
        label_dir = [label_dir]
    if not isinstance(label_dir, list):
        raise ValueError("label_dir must be a list of directories or a single directory string")

    label_files = []
    for ld in label_dir:
        if os.path.isdir(ld):
            temp = []
            for f in os.listdir(ld):
                if f.endswith('.json'):
                    temp.append(os.path.join(ld, f))
            label_files.extend(temp)

    labels = {}
    for label_file in label_files:
        timestamp = label_file.split('run_outputs_')[1].split('.json')[0]

        label_data = pd.read_json(label_file)
        labels[timestamp] = label_data
    return labels


def load_logs(log_dir):
    if not isinstance(log_dir, str): # can be str for directory or list of directories
        raise ValueError("log_dir must be a string representing the directory containing log files")
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
    logs = {}
    for log_file in log_files:
        file_path = os.path.join(log_dir, log_file)
        timestamp = log_file.split('llm_log_')[1].split('.json')[0]

        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                logs[timestamp] = data
                logs[timestamp]['model'] = logs[timestamp]['model'] if 'model' in logs[timestamp] else logs[timestamp]['hf_model']
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}", file=sys.stderr)
                continue
    return logs


def load_prompt_metadata(prompt_meta_dir):
    # map from name to prompt
    if not isinstance(prompt_meta_dir, str): # can be str for directory or list of directories
        raise ValueError("prompt_dir must be a string representing the directory containing prompt files")
    
    prompt_metadata_files = [f for f in os.listdir(prompt_meta_dir) if f.endswith('.json')]
    prompt_metadata = {}
    for prompt_metadata_file in prompt_metadata_files:
        file_path = os.path.join(prompt_meta_dir, prompt_metadata_file)

        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                prompt_metadata[prompt_metadata_file.replace('.json', '')] = data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {file_path}", file=sys.stderr)
                continue

    return prompt_metadata


def load_reference_df():
    reference_df = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/dallas_lavinia_combined_annotations.csv')
    BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/bibles/KJV.txt'

    bible_df = bible_utils.bible_helper(INPUT_PATH=BIBLE_PATH)
    text_verse_dict = {r.text.strip(): (str(r.book) + ' ' + str(r.chapter) + ':' + str(r.verse)).lower() for r in bible_df.itertuples()}

    reference_df['verse_citation'] = reference_df['King James Bible_lavinia'].apply(lambda x: text_verse_dict.get(x, None))
    reference_df['ground_truth'] = reference_df['Match_dallas']
    
    return reference_df, text_verse_dict


def load_all_annotations():
    validation_df, text_verse_dict = load_reference_df()
    human_annotations = ["bib_dallas_1-0.csv", "bib_dallas_2-0.csv", "bib_dallas_3-0.csv", "bib_lavinia_1-0.csv"]
    human_annotations = [os.path.join("/data/laviniad/", x) for x in human_annotations]

    test_annotations = []
    for human_annotation in human_annotations:
        human_df = pd.read_csv(human_annotation)
        #print("Columns: ", human_df.columns)
        #print("Head: ", human_df.head())
        human_df['verse_citation'] = human_df['King James Bible'].apply(lambda x: text_verse_dict.get(x, None))
        #human_df['ground_truth'] = human_df['Label']
        #print("Columns: ", human_df.columns)
        #print("Labels: ", human_df['Label'].unique())
        human_df['ground_truth'] = human_df['Label'] if 'Label' in human_df.columns else human_df['Match']
        human_df['ground_truth'] = human_df['ground_truth'].apply(lambda x: 1 if x == 'Match' or x == 'Other Bible Verse' else 0 if x == 'Not Match' else np.nan)
        test_annotations.append(human_df)

    # remove instances from test_annotations that are in validation_df
    test_df = pd.concat(test_annotations, ignore_index=True)
    print("Test set size before removing val duplicates: ", len(test_df))
    print("Val set size before removing test duplicates: ", len(validation_df))
    test_df = test_df[~test_df['verse_citation'].isin(validation_df['verse_citation'])]
    validation_df = validation_df[~validation_df['verse_citation'].isin(test_df['verse_citation'])]
    print("Test set size before removing NA labels: ", len(test_df))
    test_df = test_df[test_df['ground_truth'].notna()]
    print("Test set size after removing NA labels: ", len(test_df))
    # remove test_df duplicates if verse_citation, text, and ground_truth are the same
    print("Test set size before removing duplicates: ", len(test_df))
    # print sample of duplicates
    test_df = test_df.drop_duplicates(subset=['verse_citation', 'text', 'ground_truth'])
    print("Test set size after removing duplicates: ", len(test_df))
    #test_df = test_df[test_df['verse_citation'].notna()]
    #test_df = test_df[test_df['ground_truth'].notna()]
    print("Validation set size: ", len(validation_df))  

    print("Columns in validation_df: ", validation_df.columns)
    print("Columns in test_df: ", test_df.columns)              

    return validation_df, test_df


def calculate_metrics(labels, ground_truth):
    """Calculate F1 score, accuracy, precision, and recall metrics.
    
    Args:
        labels: Predicted labels
        ground_truth: True labels
        
    Returns:
        tuple: (f1, accuracy, precision, recall)
    """
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0

    f1 = f1_score(ground_truth, labels, average='weighted', zero_division=0)
    acc = accuracy_score(ground_truth, labels)
    precision = precision_score(ground_truth, labels, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, labels, average='weighted', zero_division=0)

    return f1, acc, precision, recall


def are_verses_correct(row):
    if not row['response_compliant'] and row['ground_truth'] == 0:
        return True
    elif not row['response_compliant'] and row['ground_truth'] == 1:
        return False
    
    if row['ground_truth'] == 0:
        return True
    if row['ground_truth'] == 1 and verse_correct(row['verse'], row['correct_verse_citation']):
        return True
    if row['ground_truth'] == 1 and verse_correct(row['verse'], row['correct_verse_citation']):
        return False
    return False


def evaluate_compliance_and_metrics(processed_labels, ref_df, get_baseline=True, precision_boost=False):
    """Evaluate model performance and calculate compliance metrics.
    
    Args:
        processed_labels: Dictionary containing labeled data
        ref_df: Reference dataframe with ground truth
        precision_boost: Whether to apply precision boost using Claude model
        
    Returns:
        dict: Updated processed_labels with metrics
    """
    # Ensure ground truth is integer type
    ref_df['ground_truth'] = ref_df['ground_truth'].astype(int)
    
    # Process each timestamp's labeled data
    for timestamp, label_data in tqdm(processed_labels.items(), desc="Evaluating metrics", total=len(processed_labels)):
        df = label_data['df'].copy()
        
        # Filter dataframe to only include entries in reference dataframe
        #df = df[df['text'].isin(ref_df['text'])]
        
        # Verify the dataframes have the same length
        assert len(df) == len(ref_df), "DataFrame length mismatch with reference"

        # Add ground truth and correct verse citations
        for i, row in df.iterrows():
            matching_ref = ref_df[ref_df['text'] == row['text']]
            if not matching_ref.empty:
                ref_row = matching_ref.iloc[0]
                df.at[i, 'ground_truth'] = ref_row['ground_truth']
                df.at[i, 'correct_verse_citation'] = ref_row['verse_citation']
            else:
                print(f"Warning: No matching reference for text: {row['text']}")
        
        # Mark correct predictions
        df['correct'] = df.apply(lambda row: 1 if row['label'] == row['ground_truth'] else 0, axis=1)

        # Calculate metrics for compliant responses (where both label and ground truth exist) and verse acc
        df['response_compliant'] = df['label'].notna() & df['ground_truth'].notna()
        df['verse_correct_loose'] = df.apply(are_verses_correct, axis=1)
        compliant_df = df[df['response_compliant']].copy()
        
        if not compliant_df.empty:
            f1, acc, precision, recall = calculate_metrics(
                compliant_df['label'].astype(int).tolist(),
                compliant_df['ground_truth'].astype(int).tolist()
            )
        else:
            f1, acc, precision, recall = 0.0, 0.0, 0.0, 0.0

        # Calculate metrics for all responses, replacing NaN with 0
        df['label_zeroed'] = df['label'].fillna(0).astype(int)
        df['ground_truth_zeroed'] = df['ground_truth'].fillna(0).astype(int)
        
        f1_all, acc_all, precision_all, recall_all = calculate_metrics(
            df['label_zeroed'].tolist(),
            df['ground_truth_zeroed'].tolist()
        )
        
        # Update metrics in the processed_labels dictionary
        label_data['df'] = df
        label_data['f1'] = f1
        label_data['f1_all'] = f1_all
        label_data['acc'] = acc
        label_data['acc_all'] = acc_all
        label_data['precision'] = precision
        label_data['precision_all'] = precision_all
        label_data['recall'] = recall
        label_data['recall_all'] = recall_all
        label_data['num_compliant_responses'] = len(compliant_df)
        label_data['num_responses'] = len(df)
        label_data['worked'] = len(compliant_df) > 0

    # Create baseline performance metrics
    if get_baseline:
        baseline = _create_baseline_metrics(ref_df)
        processed_labels['baseline'] = baseline

    # Apply precision boost if requested
    if precision_boost:
        pb = _apply_precision_boost(processed_labels, ref_df)
        processed_labels['baseline_with_claude'] = pb

    return processed_labels


def _create_baseline_metrics(ref_df):
    """Create baseline performance metrics.
    
    Args:
        ref_df: Reference dataframe with ground truth
        
    Returns:
        dict: Baseline metrics
    """
    baseline_df = ref_df.copy()
    baseline_df['label'] = ref_df['emb_or_ngm'].copy()
    baseline_df['verse'] = [
        ref_df['verse_citation'].iloc[i] if ref_df['emb_or_ngm'].iloc[i] else None 
        for i in range(len(ref_df))
    ]
    
    baseline_df['correct'] = baseline_df.apply(
        lambda row: 1 if int(row['label']) == int(row['ground_truth']) else 0, 
        axis=1
    )
    
    baseline_df['label_zeroed'] = baseline_df['label'].fillna(0)
    
    # Calculate metrics
    f1, acc, precision, recall = calculate_metrics(
        baseline_df['label'].astype(int).tolist(),
        ref_df['ground_truth'].astype(int).tolist()
    )
    
    return {
        'log': {"model": "baseline", "timestamp": "baseline", "prompt_metadata": None},
        'df': baseline_df,
        'f1': f1,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'num_responses': len(ref_df),
        'worked': True
    }


def _is_correct(row):
    """Check if the response is correct.
    
    Args:
        row: DataFrame row
        
    Returns:
        bool: True if correct, False otherwise
    """
    try:
        return int(row['label']) == int(row['ground_truth'])
    except ValueError:
        return False


def _apply_precision_boost(processed_labels, ref_df):
    """Apply precision boost using Claude model.
    
    Args:
        processed_labels: Dictionary containing labeled data
        ref_df: Reference dataframe with ground truth
        
    Returns:
        dict: Precision-boosted metrics
    """
    # Find Claude model results
    claude_results = next(
        (v for k, v in processed_labels.items() 
         if 'model' in v['log'] and 'claude' in v['log']['model']),
        None
    )
    
    if not claude_results:
        print("Warning: No Claude model results found for precision boost")
        return None
    
    claude_df = claude_results['df']
    claude_df['label'] = claude_df['label'].apply(bool)

    #print("Length of claude_df: ", len(claude_df))
    #print("Length of ref_df: ", len(ref_df))
    assert len(claude_df) == len(ref_df), "DataFrame length mismatch with reference"
    
    temp_df = claude_df.copy()
    temp_df.rename(columns={'label': 'llm_label', 'verse': 'llm_verse'}, inplace=True)

    # Create precision-boosted dataframe
    baseline_df = ref_df.copy()
    baseline_df.rename(columns={'emb_or_ngm': 'base_label'}, inplace=True)
    #print("Claude df: ", claude_df.head())
    #print("Baseline df: ", baseline_df.head())

    print("temp_df head")
    print(temp_df.head())

    assert(len(baseline_df) == len(temp_df))

    if 'text' in baseline_df.columns and 'text' in temp_df.columns:
        print("Merging on 'text' column")
        baseline_df = baseline_df.merge(
            temp_df[['text', 'llm_label', 'llm_verse']], 
            on='text', 
            how='left'
        )
    else:
        # If there's no common column to merge on, we need to ensure index alignment
        print("No common column for merging, using index-based assignment")
        baseline_df['llm_label'] = temp_df['llm_label'].values
        baseline_df['llm_verse'] = temp_df['llm_verse'].values

    print("Baseline df: ", baseline_df.head())
    
    # Apply precision boost logic
    baseline_df['label'] = baseline_df.apply(
        lambda row: bool(row['base_label'] and row['llm_label']), 
        axis=1
    )
    
    baseline_df['verse'] = baseline_df['llm_verse'].copy()
    baseline_df['correct'] = baseline_df.apply(
        _is_correct, 
        axis=1
    )
    
    baseline_df['label_zeroed'] = baseline_df['label'].astype(int)
    
    # Prepare dataframe
    print("Number of NAs in baseline_df ground truth: ", baseline_df['ground_truth'].isna().sum())
    print("Rows with NAs in baseline_df: ", baseline_df[baseline_df['ground_truth'].isna()])
    baseline_df = baseline_df[baseline_df['label'].notna() & baseline_df['ground_truth'].notna()]
    baseline_df['ground_truth'] = baseline_df['ground_truth'].astype(int)
    baseline_df['label'] = baseline_df['label'].astype(int)

    # Calculate metrics
    f1, acc, precision, recall = calculate_metrics(
        baseline_df['label'].tolist(),
        baseline_df['ground_truth'].tolist()
    )
    
    return {
        'log': {"model": "baseline_with_claude", "timestamp": "baseline_with_claude", "prompt_metadata": None},
        'df': baseline_df,
        'f1': f1,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'num_responses': len(baseline_df),
        'worked': True
    }
