## TODO: convert from placeholder code to functional
import pandas as pd
import json
from tqdm import tqdm
import yaml

OUTPUT_PATH = "/data/laviniad/congress_errata/bioguide_to_gender.json"
congress_member_path = "/data/laviniad/congresspeople_data/"
congress_member_files = ["legislators-current.yaml", "legislators-historical.yaml"]

# load data
print("Loading current data...")
with open(congress_member_path + congress_member_files[0], 'r') as file:
    current_data = yaml.safe_load(file)

print("Loading historical data...")
with open(congress_member_path + congress_member_files[1], 'r') as file:
    historical_data = yaml.safe_load(file)

print(current_data[0]) # debug/get sense of structure

# map bioguide_id to gender
print("Mapping bioguide ID to gender...")
bio_to_gender = {}
for member in tqdm(current_data):
    bio_id = member['id']['bioguide']
    gender = member['bio']['gender']
    bio_to_gender[bio_id] = gender

for member in tqdm(historical_data):
    bio_id = member['id']['bioguide']
    gender = member['bio']['gender']
    if bio_id not in bio_to_gender.keys():
        bio_to_gender[bio_id] = gender

# dump to json
print("Dumping JSON...")
with open(OUTPUT_PATH, 'w') as file:
    json.dump(bio_to_gender, file)
    print(f"Dumped bio-gender dict to {OUTPUT_PATH}")
