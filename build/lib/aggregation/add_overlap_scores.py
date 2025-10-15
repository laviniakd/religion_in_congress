# imports
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

import spacy
from spacy.pipeline import Sentencizer

nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
nlp.add_pipe("sentencizer", config=config)
from spacy.lang.en import stop_words
import argparse
stop_words = stop_words.STOP_WORDS

DEBUG = False
add_topic = True
USE_ONLY_GOD_TALK = False
ADD_KEYWORD_BOOLS = True
REMOVE_TEXT = False

# constants
# Argument parser
parser = argparse.ArgumentParser(description="Process congressional data.")
parser.add_argument('--congress_path', type=str, default="/data/corpora/congressional-record/", help='Path to the congressional data.')
parser.add_argument('--output_prefix', type=str, default="/data/laviniad/congress_errata/congress_df", help='Prefix for the output file.')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with a sample of the data.')
parser.add_argument('--add_topic', action='store_true', help='Add topic labels to the data.')
parser.add_argument('--use_only_god_talk', action='store_true', help='Use only god talk keywords.')
parser.add_argument('--add_keyword_bools', action='store_true', help='Add boolean columns for specific keywords.')
parser.add_argument('--remove_text', action='store_true', help='Remove the text column from the final dataframe.')

args = parser.parse_args()

# Constants
congress_path = args.congress_path
output_prefix = args.output_prefix

DEBUG = args.debug
add_topic = args.add_topic
USE_ONLY_GOD_TALK = args.use_only_god_talk
ADD_KEYWORD_BOOLS = args.add_keyword_bools
REMOVE_TEXT = args.remove_text

if USE_ONLY_GOD_TALK:
    limited_kws = get_full_keywords(ONLY_GOD_TALK=True)

# data loading
print("***** Getting bioguide data *****")
bio_dict = load_bios()
bio_to_dw = load_bioguide_id_to_dw_nominate()


for k in tqdm(bio_dict.keys()):
    if k in bio_to_dw.keys():
        dw = bio_to_dw[k]
        bio_dict[k]['dw_nominate'] = dw
    else:
        bio_dict[k]['dw_nominate'] = [None, None]


print("***** Getting state-level and county-level religion data *****")
states = get_states() # returns full name: abbreviation dictionary

county_data = pd.read_csv('/home/laviniad/projects/CausalMLProjectData/data/county_religion.csv')

def get_state(x):
    if x in states.keys():
        return states[x]
    return 'NONE'

def get_county_code_with_state(row):
    state_code = str(row['STATE'])
    county_name = str(row['County Name']) # if NaN, not going to index anyways
    county_name = county_name.upper().replace(' COUNTY', '').replace(' PARISH', '')

    return county_name + ', ' + state_code
    
print("Loading county and state data")
county_data['STATE'] = county_data['State Name'].apply(get_state).apply(lambda x: x.upper())
county_data['adherent_pop'] = county_data['Adherents'].apply(lambda x: float(str(x).replace(',', '')))
county_data['total_pop_2020'] = county_data['2020 Population'].apply(lambda x: float(str(x).replace(',', '')))
state_data = county_data.groupby('STATE').agg('sum').reset_index() # can only really use adherent_pop and total_pop_2020 after this

print("Getting adherent percentage stats")
state_data['perc_adherent_2020'] = state_data['adherent_pop'] / state_data['total_pop_2020']
county_data['county_code'] = county_data.apply(get_county_code_with_state, axis=1)
county_data['perc_adherent_2020'] = county_data['adherent_pop'] / county_data['total_pop_2020']
county_to_adherent_pop = {k: v for k,v in zip(county_data['county_code'], county_data['adherent_pop'])}
county_to_perc_adherents = {k: v for k,v in zip(county_data['county_code'], county_data['perc_adherent_2020'])}

print("Getting bioguide --> county mapping")
with open('/data/laviniad/congress_errata/bioguide_to_county.json', 'r') as f:
    bio_to_county = json.load(f)

def get_enclosing_county(row):
    if row['congress_num'] != 118:
        return 'NOT_118'

    if row['is_in_senate']:
        return 'IN_SENATE'
    if row['bio_id'] in bio_to_county.keys():
        return bio_to_county[row['bio_id']]
    
    return 'NO_DATA'


print("***** Getting other state demographic data *****")
state_2000to2010_path = '/home/laviniad/projects/religion_in_congress/data/state_stats/2000to2010.csv'
state_2010to2020_path = '/home/laviniad/projects/religion_in_congress/data/state_stats/2010to2020.csv'
state_2000to2010 = pd.read_csv(state_2000to2010_path)
state_2010to2020 = pd.read_csv(state_2010to2020_path)

print("***** Loading congressional data and speaker covariates *****")
congress_df = load_full_df_from_raw(congress_path, remove_procedural_speeches=True)
if DEBUG:
    congress_df = congress_df.sample(100)

congress_df = induce_gender(congress_df)
congress_df = induce_party_and_state(congress_df)
congress_df['is_in_senate'] = congress_df['chamber'] != 'House'

if add_topic:
    print("Labeling topic")
    congress_df = induce_topic(congress_df)
    congress_df['is_abortion'] = congress_df['main_topic'] == 'Abortion'
    congress_df['is_immigration'] = congress_df['main_topic'] == 'ImmigrationBorder'
    congress_df['is_christianity'] = congress_df['main_topic'] == 'Christianity'
    congress_df['is_science_technology'] = congress_df['main_topic'] == 'ScienceTechnology'
    congress_df['is_health_insurance'] = congress_df['main_topic'] == 'HealthInsurance'

print("Labeling county and county's percent that are religious")
congress_df['enclosing_county'] = congress_df.apply(get_enclosing_county, axis=1)
congress_df['perc_adherents'] = congress_df['enclosing_county'].apply(lambda x: county_to_perc_adherents[x] if x in county_to_perc_adherents.keys() else None)

print("Labeling religion, chamber, and party")
congress_df = induce_religion_if_118th(congress_df)
congress_df['is_republican'] = congress_df['party'] == 'Republican'

if not DEBUG:
    assert(2017 in congress_df['year'].unique()) # check that data loading is working like it should

    
def church_in_bio(x):
    if x in bio_dict.keys():
        return 'church' in bio_dict[x]['profileText']
    else:
        return 'unknown'
    
def get_dw_nominate(x):
    if x in bio_dict.keys():
        return bio_dict[x]['dw_nominate'][0], bio_dict[x]['dw_nominate'][1]

    return None, None


print("***** Adding party, state, and DW-NOMINATE covariates to congress_df *****")
congress_df['church_in_bio'] = congress_df['bio_id'].progress_apply(church_in_bio)
congress_df[['dw_nom_1', 'dw_nom_2']] = congress_df['bio_id'].progress_apply(lambda x: pd.Series(get_dw_nominate(x)))

# can't identify speaker, basically
def not_nonpartisan_speaker(name):
    indicative_strings = ['pro tempore', 'PRESIDING OFFICER',
                         'The CHAIR', 'The Acting CHAIR', 'The SPEAKER']
    for s in indicative_strings:
        if s in name:
            return False
    
    return True

congress_df_filtered = congress_df[congress_df['speaker'].apply(not_nonpartisan_speaker)]

print("***** Keyword overlap *****")

if USE_ONLY_GOD_TALK:
    objs = congress_df['text'].progress_apply(lambda x: get_typed_overlap(x, keywords=limited_kws))
    congress_df['lexical'] = [o[0] for o in objs]
    congress_df['num_general'] = [o[1]['general'] for o in objs]
    congress_df['num_christian'] = [o[1]['christian'] for o in objs]
    congress_df['length'] = [o[2] for o in objs]
else:
    objs = congress_df['text'].progress_apply(lambda x: get_typed_overlap(x))
    congress_df['lexical'] = [o[0] for o in objs]
    congress_df['num_general'] = [o[1]['general'] for o in objs]
    congress_df['num_christian'] = [o[1]['christian'] for o in objs]
    congress_df['length'] = [o[2] for o in objs]
congress_df['binary_lex'] = congress_df['lexical'].progress_apply(lambda x: x > 0)
state_reverse = {v: k for k, v in states.items()} # code to full
congress_df['full_state'] = congress_df['state'].apply(lambda x: state_reverse[x] if x in state_reverse.keys() else None)

print("***** Adding demographic covariates to data *****")

print("Getting demographic data")

state_to_year_to_pwhite = {s: {} for s in states.keys()}
state_to_year_to_pblack = {s: {} for s in states.keys()}

year_range = range(2000, 2021)
for state in tqdm(states):
    for year in year_range:
        if year > 2009:
            demo_csv = state_2010to2020
        else:
            demo_csv = state_2000to2010
        pwhite, pblack = map_state_year_to_demographics(state, year, demo_csv)
        state_to_year_to_pwhite[state][year] = pwhite
        state_to_year_to_pblack[state][year] = pblack

pwhite_series = []
for idx, row in tqdm(congress_df.iterrows(), total=len(congress_df.index)):
    state = row['full_state']
    year = row['year']
    if not (year in year_range):
        pwhite_series.append('not_in_range')
    elif state is not None:
        pwhite_series.append(state_to_year_to_pwhite[state][year])
    else:
        pwhite_series.append(None)

congress_df['state_perc_white'] = pwhite_series

pblack_series = []
for idx, row in tqdm(congress_df.iterrows(), total=len(congress_df.index)):
    state = row['full_state']
    year = row['year']
    if not (year in year_range):
        pblack_series.append('not_in_range')
    elif state is not None:
        pblack_series.append(state_to_year_to_pblack[state][year])
    else:
        pblack_series.append(None)

congress_df['state_perc_black'] = pblack_series

if USE_ONLY_GOD_TALK:
    output_prefix += '_limited'

# quick and dirty presence labeling
if ADD_KEYWORD_BOOLS:
    congress_df['god'] = congress_df['text'].str.contains('god', case=False, na=False)
    congress_df['bible'] = congress_df['text'].str.contains('bible', case=False, na=False)
    congress_df['jesus'] = congress_df['text'].str.contains('jesus', case=False, na=False)
    congress_df['faith'] = congress_df['text'].str.contains('faith', case=False, na=False)
    congress_df['pray'] = congress_df['text'].str.contains('pray', case=False, na=False)

if REMOVE_TEXT:
    congress_df.drop('text', axis=1, inplace=True)

congress_df.to_json(output_prefix + '.json')
print(f"Dumped to {output_prefix}.json")
print("***** Done *****")
