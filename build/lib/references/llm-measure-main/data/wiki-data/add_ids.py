import json
from tqdm import tqdm

def read_json(input_filename):
    with open(input_filename) as input_file:
        data = json.load(input_file)
    return data


def main():
    all_data_path = "/data/madesai/history-llm-data/wikipedia-json-files/all_wiki.json"
    all_data = read_json(all_data_path)

    count = 0 
    for year in tqdm(range(1,2020)):
       
        year_file_path = "/data/madesai/history-llm-data/wikipedia-json-files/"+str(year)+"_births.json"
        try:
            year_file = read_json(year_file_path)
        except:
            print(year)
        for k in year_file:
            year_file[k]['id'] = count
            if k not in all_data:
                all_data[k] = year_file[k]
            else:
                all_data[k]['id'] = count
            count += 1
        with open(year_file_path, "w") as f:
            json.dump(year_file,f)

    with open(all_data_path, "w") as f:
        json.dump(all_data,f)

if __name__ == "__main__":
    main()