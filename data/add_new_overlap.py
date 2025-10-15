import pandas as pd

from data.congress_utils import load_full_df_from_raw, induce_party_and_state, induce_topic, load_bioguide_id_to_dw_nominate, map_state_year_to_demographics, load_bios, induce_religion_if_118th, induce_gender
from data.data_utils import load_spacy_sentencizer, get_full_keywords, get_simple_overlap, get_relig_boolean, get_states, get_typed_overlap
import numpy as np
from pprint import pprint
import string
from collections import Counter
import nltk
import json
from tqdm import tqdm
tqdm.pandas()

output_prefix = "/data/laviniad/congress_errata/congress_df"
congress_df = pd.read_json(output_prefix + '.json')

# now loading in correct KWs
objs = congress_df['text'].progress_apply(lambda x: get_typed_overlap(x))
print("loaded kws")
exit()
congress_df['old_lexical'] = congress_df['lexical']
congress_df['old_num_general'] = congress_df['num_general']
congress_df['old_num_christian'] = congress_df['num_christian']
congress_df['old_binary_lex'] = congress_df['binary_lex']

congress_df['lexical'] = [o[0] for o in objs]
congress_df['num_general'] = [o[1]['general'] for o in objs]
congress_df['num_christian'] = [o[1]['christian'] for o in objs]
congress_df['length'] = [o[2] for o in objs]
congress_df['binary_lex'] = congress_df['lexical'].progress_apply(lambda x: x > 0)

print("Now dumping")
congress_df.to_json(output_prefix + '.json')
print(f"Dumped to {output_prefix}.json")