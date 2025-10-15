import json
from tqdm import tqdm
from data.congress_utils import load_full_df_from_raw, induce_party_and_state

output_path = "/data/laviniad/congress_errata/only_nonprocedural_congressional_speeches.jsonlist"
nonproc_path = "/data/laviniad/congress_errata/nonprocedural_indices.json"
congress_df = load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True, nonprocedural_indices_path=nonproc_path)
congress_df = induce_party_and_state(congress_df)
congress_dict_records = congress_df.to_dict(orient='records')

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in tqdm(data):
            json_line = json.dumps(item)
            file.write(json_line + '\n')

write_jsonl(output_path, congress_dict_records)
