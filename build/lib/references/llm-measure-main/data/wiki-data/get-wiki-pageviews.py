import json
import random
import re
import wikipediaapi
import re
import requests
from tqdm import tqdm
import seaborn as sns
import argparse


def initiate_request():
    global wiki_wiki
    wiki_wiki = wikipediaapi.Wikipedia('GenerateText (madesai@umich.edu)', 'en')
    return wiki_wiki


def get_page_views(page):
    try:
        name = page.fullurl.removeprefix("https://en.wikipedia.org/wiki/")
    except:
        name = page.title.replace(" ", "_")
    address = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/" + name + "/monthly/2015010100/2023083100"
    headers = {'User-Agent': 'GenerateText (madesai@umich.edu)'}

    resp = requests.get(address, headers=headers, timeout=20.0)
    if resp.ok:
        details = resp.json()
        total_views = 0
        for month in details['items']:
            total_views += month['views']
        return total_views
    else:
         return None
    

def read_json(input_filename):
    with open(input_filename) as input_file:
        data = json.load(input_file)
    return data


def add_page_views_to_file(jsonfile, wiki_wiki):
    data = read_json(jsonfile)
    for name in tqdm(data):
        name_page = wiki_wiki.page(name)
        page_views = get_page_views(name_page)
        data[name]['page_views'] = page_views
    return data

def parse_arguments():
    parser = argparse.ArgumentParser(description="llm-measure")
    parser.add_argument("--year",
                        type = str,)
    args = parser.parse_args()
    return args

def main():
    wiki_wiki = initiate_request()
    args = parse_arguments()
    year = args.year
    jsonfile = "/data/madesai/history-llm-data/wikipedia-json-files/"+str(year)+"_births.json"
    print(jsonfile)
    page_views = add_page_views_to_file(jsonfile, wiki_wiki)

    with open(jsonfile, "w") as f:
        json.dump(page_views,f)

if __name__ == "__main__":
    main()