import json
import random
import re
import wikipediaapi
import re
import requests
from tqdm import tqdm
import seaborn as sns
import argparse

def read_json(input_filename):
    with open(input_filename, encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data

def parse_arguments():
    parser = argparse.ArgumentParser(description="make-wiki-sample")
    parser.add_argument("--nsamples",
                        type = int,)
    parser.add_argument("--views",
                        type=int,
                        default=0)
    parser.add_argument("--outfilename",
                        type= str)
    parser.add_argument("--name_length",
                        type = int,
                        default=None)
    parser.add_argument("--excl_crit",
                        type=str,
                        default=None,
                        help="regex string. will be used to exclude names")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    nsamples = args.nsamples
    views = args.views
    outfile = args.outfilename

    name_length = args.name_length
    exclude = args.excl_crit

    all_data = {"ids":{},"birth_year":{},"people":{},"page_views":{}}
    for year in range(1940,1999):
        year_data = read_json("/data/madesai/history-llm-data/wikipedia-json-files/"+str(year)+"_births.json")
        if views: 
            candidates = []
            for page in year_data:
            #    print(type(year_data[page]['page_views']), type(views))
                
                try:
                    if name_length and len(page.split(" ")) > name_length:
                        continue
                    elif exclude and re.findall(exclude,page):
                            continue
                    elif year_data[page]['page_views'] and year_data[page]['page_views'] > views:
                        candidates.append(page)
                except:
                    print(year_data[page], year)
        else: 
            candidates = list(year_data)

        if len(candidates) >= nsamples:
            keys = random.sample(candidates, nsamples)
        else:
            print("Less than {} pages with more than {} views in {}.".format(nsamples,views,year))
            keys = candidates
        
        for k in keys:
            all_data[k] = year_data[k]
            all_data["ids"][year_data[k]['id']] = year_data[k]['id']
            all_data["birth_year"][year_data[k]['id']] = [year_data[k]['birth_year']]
            all_data["page_views"][year_data[k]['id']] = [year_data[k]['page_views']]
            all_data["people"][year_data[k]['id']] = k
        
    outpath = "/home/madesai/llm-measure/data/wiki-data/files/"+outfile
    with open(outpath, "w") as f:
        json.dump(all_data,f)

if __name__ == "__main__":
    main()
