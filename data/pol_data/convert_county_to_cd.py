import pandas as pd
import json
import os
from tqdm import tqdm

political_data_path = "/home/laviniad/projects/religion_in_congress/data/pol_data/"
output_path = "/data/laviniad/congress_errata/cd_to_county.json"

print("Loading data")
county_to_fips = pd.read_csv(political_data_path + "countypres_2000-2020.csv", dtype={'county_fips': str, 'state_po': str, 'county_name': str})
county_to_fips = {row['county_name'] + ", " + row['state_po']: row['county_fips'] for idx, row in county_to_fips[county_to_fips["year"] == 2020].iterrows()}
fips_to_county = {v: k for k, v in county_to_fips.items()}

county_to_zip = pd.read_csv(political_data_path + "zip_code_database.csv", dtype={'zip': str, 'county': str})
zip_to_county = pd.read_csv(political_data_path + "zip_county_122023.csv", dtype={'ZIP': str, 'COUNTY': str})

fips_to_state = {str(row['COUNTY'])[:2]: row['USPS_ZIP_PREF_STATE'] for idx, row in zip_to_county.iterrows()}

def get_county_str(row):
    if isinstance(row['county'], str):
        return row['county'] + ", " + row['state']
    
    return 'unknown'

zip_to_county['county'] = zip_to_county['COUNTY'].apply(lambda x: fips_to_county[x] if x in fips_to_county.keys() else None) # convert FIPS codes to county names
zip_to_cd = pd.read_csv(political_data_path + "zip_cd_122023.csv", dtype={'ZIP': str, 'CD': str})
zip_to_cd_117 = pd.read_csv(political_data_path + "zip_cd_122021.csv", dtype={'zip': str, 'cd': str})

print("Merging data")
cd_to_county = {}
state_county_mismatch = []
if not os.path.exists(output_path):
    for idx, row in tqdm(zip_to_cd.iterrows(), total=len(zip_to_cd.index)):
        #print(row)
        cd = row['CD']
        cd_num = cd[-2:] # last two digits
        cd_state = fips_to_state["0" + cd[0]] if len(cd[:-2]) == 1 else str(fips_to_state[cd[:-2]])

        cd = cd_state + "-" + cd_num

        county = zip_to_county[zip_to_county["ZIP"] == row["ZIP"]]["county"].values[0]
        if county is not None:
            if county[-2:] != cd_state:
                state_county_mismatch.append((county, cd_state))

            cd_to_county[cd] = county

    print(f"Number of mismatched counties and states: {len(set(state_county_mismatch))}")
    print(f"First five mismatches: {list(set(state_county_mismatch))[:5]}")
    print(f"Number of matched counties: {len(cd_to_county)}")

    with open(output_path, "w") as f:
        json.dump(cd_to_county, f)
        print(f'Dumped to {f.name}')
else:
    with open(output_path, "r") as f:
        cd_to_county = json.load(f)
        print(f'Dumped to {f.name}')

cd_to_county_117 = {}
state_county_mismatch = []
for idx, row in tqdm(zip_to_cd_117.iterrows(), total=len(zip_to_cd_117.index)):
    #print(row)
    cd = row['cd']
    zip = row['zip']
    cd_num = cd[-2:] # last two digits
    cd_state = fips_to_state["0" + cd[0]] if len(cd[:-2]) == 1 else str(fips_to_state[cd[:-2]])
    cd = cd_state + "-" + cd_num

    if cd not in cd_to_county_117.keys():
        if zip in zip_to_county["ZIP"].values:
            county = zip_to_county[zip_to_county["ZIP"] == zip]["county"].values[0]
            if county is not None:
                if county[-2:] != cd_state:
                    state_county_mismatch.append((county, cd_state))

                cd_to_county_117[cd] = county
        else:
            print(f"ZIP code {zip} not in zip_to_county dictionary")

print(f"Number of mismatched counties and states: {len(set(state_county_mismatch))}")
print(f"First five mismatches: {list(set(state_county_mismatch))[:5]}")
print(f"Number of matched counties (117th): {len(cd_to_county_117)}")

print("Now iterating through 118th+117th Congress members and labeling with counties")
bioguide_to_county = {}
output_path = "/data/laviniad/congress_errata/bioguide_to_county.json"
bioguide_path = political_data_path + "legislators-current.json"


with open(bioguide_path, "r") as f:
    congress_people = json.load(f)

    for obj in tqdm(congress_people):
        most_recent_term = obj['terms'][-1]
        bioguide = obj['id']['bioguide']
        state = most_recent_term['state']
        if 'district' in most_recent_term.keys() and most_recent_term["start"].startswith("2023"): # 118th
            print("found 118th term")
            cd = str(most_recent_term['district'])
            # cds are in 0[digit] format is < 10
            cd = "0" + cd if len(cd) == 1 else cd
            cd = state + "-" + cd

            try:
                bioguide_to_county[bioguide] = cd_to_county[cd]
            except KeyError:
                print(f"Could not find {cd} in cd_to_county")
                print(f"Name: {obj['name']}")


print("Now on 117th...")
with open(bioguide_path, "r") as f:
    congress_people = json.load(f)

    for obj in tqdm(congress_people):
        most_recent_term = obj['terms'][-1]
        second_most = None
        if len(obj['terms']) > 1:
            second_most = obj['terms'][-2] # can't be third most or earlier due to arrow of time, it's 2024 rn (this is a little brittle)
        bioguide = obj['id']['bioguide']
        state = most_recent_term['state']
        if 'district' in most_recent_term.keys() and most_recent_term["start"].startswith("2021"): # 117th
            print("found 117th term")
            cd = str(most_recent_term['district'])
            # cds are in 0[digit] format is < 10
            cd = "0" + cd if len(cd) == 1 else cd
            cd = state + "-" + cd

            try:
                bioguide_to_county[bioguide] = cd_to_county_117[cd]
            except KeyError:
                print(f"Could not find {cd} in cd_to_county")
                print(f"Name: {obj['name']}")
        elif (second_most != None) and 'district' in second_most.keys() and second_most["start"].startswith("2021"): # 117th
            print("found 117th term")
            cd = str(second_most['district'])
            # cds are in 0[digit] format is < 10
            cd = "0" + cd if len(cd) == 1 else cd
            cd = state + "-" + cd

            try:
                bioguide_to_county[bioguide] = cd_to_county_117[cd]
            except KeyError:
                print(f"Could not find {cd} in cd_to_county")
                print(f"Name: {obj['name']}")



with open(output_path, "w") as f:
    json.dump(bioguide_to_county, f)
    print(f'Dumped to {f.name}')

