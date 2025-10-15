import json
import argparse

NUM = 150
PROJECT_PATH = '/home/laviniad/projects/religion_in_congress/'
INPUT_PATH = '/data/laviniad/sermons-ir/log_odds/final/congress_sermon_odds_FINAL.json'
OUTPUT_PATH = PROJECT_PATH + 'src/keywords/keywords_from_congress_FINAL.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--num', default=150, type=int)
parser.add_argument('--input_path', default=INPUT_PATH)
parser.add_argument('--output_path', default=OUTPUT_PATH)

args = parser.parse_args()

NUM = args.num
INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path

with open(INPUT_PATH) as f:
    data = json.load(f)

# check
for k, v in data.items():
    if k[0].isupper():
        print(k)
    
word_list = list(data)[:NUM]

with open(OUTPUT_PATH, 'w') as f:
    f.writelines([w + '\n' for w in word_list])
