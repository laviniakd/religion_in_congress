import little_mallet_wrapper as lmw
from data import congress_utils
from data.data_utils import get_full_keywords
import pandas as pd
import argparse
import sys

path_to_mallet = '/home/laviniad/mallet-2.0.8/bin/mallet'
#MIN_UTTERANCE_LENGTH = 100
#SAMPLE_DOC_NUM = 50000

lmw_stops = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 'amp']

from data.bible_utils import bible_helper

parser = argparse.ArgumentParser()
parser.add_argument('--num_topics', default=100, type=int)
parser.add_argument('--sample', default=-1, type=int)
parser.add_argument('--load_indices', action='store_true')
parser.add_argument('--remove_keywords', action='store_true')
args = parser.parse_args()

if args.load_indices:
    nonproc_path = "/data/laviniad/congress_errata/nonprocedural_indices.json"
    congress_df = congress_utils.load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True, nonprocedural_indices_path=nonproc_path)
else:
    congress_df = congress_utils.load_full_df_from_raw('/data/corpora/congressional-record/', remove_procedural_speeches=True)
    congress_df.to_csv('/data/laviniad/congress_errata/congress_df.csv', index=True)
    print("Dumped congress_df to /data/laviniad/congress_errata/congress_df.csv")

#congress_df = pd.read_csv('/data/laviniad/congress_errata/congress_df.csv') # already filtered
if args.sample > 0:
    congress_df = congress_df.sample(args.sample)
print("Length of congressional data: " + str(len(congress_df.index)))

congress_df.to_csv('/data/laviniad/congress_errata/small_congress.csv', index=True)

if args.remove_keywords:
    print("Loading keywords to mask out")
    stop_list = [o.lower() for o in get_full_keywords()]

training_data = [lmw.process_string(str(t), stop_words=lmw_stops + stop_list) for t in congress_df['text'].tolist()]
print(f"Size of training data: {sys.getsizeof(training_data)}")
#training_data = [d for d in training_data if d.strip()]

len(training_data)
lmw.print_dataset_stats(training_data)

output_directory_path = '/data/laviniad/sermons-ir/topic_models/congress/'

topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet,
                                                              output_directory_path,
                                                              args.num_topics,
                                                              training_data)

for i, t in enumerate(topic_keys):
    print(i, '\t', ' '.join(t[:10]))

for p, d in lmw.get_top_docs(training_data, topic_distributions, topic_index=0, n=3):
    print(round(p, 4), d)
    print()
