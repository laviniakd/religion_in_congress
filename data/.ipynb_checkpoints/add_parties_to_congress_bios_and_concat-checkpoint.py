import json
from pprint import pprint
import os
    

def extract_and_add_party(input_directory, output_file):
    json_files = [f for f in os.listdir(input_directory) if f.endswith('.json')]
    unmatched_party_counter = 0
    
    with open(output_file, 'w') as output:
        for json_file in json_files:
            file_path = os.path.join(input_directory, json_file)
            with open(file_path, 'r') as input_file:
                data = json.load(input_file)

                party_affiliation = None
                if 'jobPositions' in data and data['jobPositions']:
                    congress_entry = data['jobPositions'][-1]['congressAffiliation']
                    congress_type = congress_entry['congress']['congressType']
                    if congress_type != 'ContinentalCongress' and congress_type != 'ConfederationCongress': 
                        # indicates no party
                        party_affiliation = None
                        try:
                            if 'partyAffiliation' in congress_entry.keys():
                                party_affiliation = congress_entry['partyAffiliation'][0]['party']['name']
                            else:
                                for c in data['jobPositions']:
                                    if 'partyAffiliation' in c['congressAffiliation'].keys():
                                        party_affiliation = c['congressAffiliation']['partyAffiliation'][0]['party']['name']
                                if party_affiliation == None:
                                    raise(KeyError)
                        except KeyError:
                            print(f"Unknown party for {data['givenName']} {data['familyName']}, data is: ")
                            #print(json.dumps(data, indent=4))
                            unmatched_party_counter += 1
                            #print(f"Instead data['jobPositions'][0]['congressAffiliation'] is {congress_entry}")

                data['party'] = party_affiliation

                json_line = json.dumps(data) + '\n'
                output.write(json_line)
        print(f"unmatched bios: {unmatched_party_counter}, or {unmatched_party_counter / len(data)}% of all entries")

if __name__ == "__main__":
    input_directory = "/data/laviniad/congress_bioguides/"
    output_file = "/data/laviniad/congress_bioguides.jsonlist"
    extract_and_add_party(input_directory, output_file)
