"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
basic idea: we have a bunch of different measures and we want to aggregate them into one dataframe

the format is still congressional record document-level, but each sentence is associated with a separate score; 
i.e., the [measure]_scores columns are lists pertaining to each sentence instead of single values
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## imports
import pandas as pd
from data.congress_utils import load_full_df_from_raw, induce_party_and_state, induce_topic
from data.data_utils import get_simple_overlap, get_relig_boolean
from tqdm import tqdm
import json

tqdm.pandas()

## constants
output_path = '/data/laviniad/sermons-ir/unified_congress_df_with_scores.json'
classifier_path = '/data/laviniad/congress_errata/idx_to_classifier_output.json'
bible_to_base_path = '/data/laviniad/congress_errata/idx_to_ppl_bible_and_base.json'
embedding_results = '/data/laviniad/sermons-ir/modeling/mpnet_results/results.csv'

DEBUG=True

## load data
print("Loading data")
congress_df = load_full_df_from_raw('/data/corpora/congressional-record', remove_procedural_speeches=True)
congress_df = induce_party_and_state(congress_df)
congress_df = induce_topic(congress_df)

if DEBUG:
    congress_df = congress_df.sample(100) # ensure it runs

## keyword overlap
print("Computing keyword overlap and relig boolean")
congress_df['lexical_overlap'] = congress_df['text'].progress_apply(get_simple_overlap)
congress_df['has_relig_keyword'] = congress_df['lexical_overlap'].apply(lambda x: x > 0)

## classifier output
print("Getting classifier probability")
with open(classifier_path, 'r') as f:
    idx_to_classifier = json.load(f)
    
def get_prob_religious_list(idx):
    results = idx_to_classifier[idx]
    r_list = []
    # 1 is sermon, 0 is congress
    
    for r in results:
        if isinstance(r, dict):
            r_list.append(r['probs'][1])
        else:
            r_list.append(0)
    
    return r_list


def get_religious_label_list(idx):
    results = idx_to_classifier[idx]
    r_list = []
    # 1 is sermon, 0 is congress
    
    for r in results:
        if isinstance(r, dict):
            r_list.append(r['label'])
        else:
            r_list.append('unknown')
    
    return r_list
            
    
congress_df['classifier_probability_list'] = congress_df.index.to_series().apply(get_prob_religious_list)
congress_df['classifier_label'] = congress_df.index.to_series().apply(get_religious_label_list)

## embedding reference cosine similarity score
print("Getting max similar embeddings for verse reference")
congress_df['official_index'] = congress_df.index.to_series()
mpnet_scores = pd.read_csv(embedding_results) # columns are congress_idx,text,most_similar_verse,cosine_similarity,verse_citation
mpnet_scores['congress_idx'] == mpnet_scores['congress_idx'].apply(lambda x: x.replace('tensor(', '').replace(')', '')).apply(int)

def get_max_cs(index_num, df):
    relevant_rows = df[df['congress_idx'] == index_num]
    if len(relevant_rows) > 0:
        idx_of_verse = relevant_rows['cosine_similarity'].idxmax()
        cs, citation, text = relevant_rows.loc[idx_of_verse, ['cosine_similarity', 'verse_citation', 'text']]
        return cs
    else:
        val = -1

        
def get_max_citation(index_num, df):
    relevant_rows = df[df['congress_idx'] == index_num]
    if len(relevant_rows) > 0:
        idx_of_verse = relevant_rows['cosine_similarity'].idxmax()
        cs, citation, text = relevant_rows.loc[idx_of_verse, ['cosine_similarity', 'verse_citation', 'text']]
        return citation
    else:
        val = ''
        

def get_max_text(index_num, df):
    relevant_rows = df[df['congress_idx'] == index_num]
    if len(relevant_rows) > 0:
        idx_of_verse = relevant_rows['cosine_similarity'].idxmax()
        cs, citation, text = relevant_rows.loc[idx_of_verse, ['cosine_similarity', 'verse_citation', 'text']]
        return text
    else:
        val = ''

congress_df['max_cosine_sim_ref'] = congress_df.index.to_series().apply(lambda x: get_max_cs(x, mpnet_scores))
congress_df['verse_citation_ref'] = congress_df.index.to_series().apply(lambda x: get_max_citation(x, mpnet_scores))
congress_df['og_text_ref'] = congress_df.index.to_series().apply(lambda x: get_max_text(x, mpnet_scores))


## bible-to-base ppl ratio
print("Getting Bible/base PPLs")
with open(bible_to_base_path, 'r') as f:
    idx_to_ppl_dict = json.load(f) # should be idx --> {'base': ppl wrt base model, 'bible': ppl wrt bible model}

congress_df['bible_ppl'] = congress_df.index.map(lambda x: idx_to_ppl_dict[int(x)]['bible'])
congress_df['base_ppl'] = congress_df.index.map(lambda x: idx_to_ppl_dict[int(x)]['base'])
congress_df['bible_over_base'] = congress_df['bible_ppl'] / congress_df['base_ppl'] # i.e., normalized bible ppl

## topic model score wrt christianity quantity
print("Getting if church topic")
congress_df['is_church_topic'] = (congress_df['main_topic'] == 'Church')

## output as json; will retain list structure better
if not DEBUG:
    congress_df.to_json(output_path)
    print(f'Just dumped unified dataframe to {output_path}')
else:
    print('Did not dump, since debugging')
