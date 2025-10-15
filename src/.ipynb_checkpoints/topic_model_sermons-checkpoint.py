import little_mallet_wrapper as lmw
import pandas as pd

# TODO: start running topic model

path_to_mallet = '/shared/0/resources/mallet/mallet-2.0.8/bin/mallet'
NUM_TOPICS = 150
MIN_UTTERANCE_LENGTH = 100
SAMPLE_DOC_NUM = 50000

from data.bible_utils import bible_helper

MED_PATH = '/shared/3/projects/sermons-ir/sermoncentral/with_columns.csv'
sermon_df = pd.read_csv(MED_PATH)

# load data???
# sermon_df = pd.concat(sermon_df).sample(SAMPLE_DOC_NUM)
print(sermon_df.head())
print("Length of sermon_df: " + str(len(sermon_df.index)))

training_data = [lmw.process_string(str(t)) for t in sermon_df['text'].tolist()]
training_data = [d for d in training_data if d.strip()]

len(training_data)
lmw.print_dataset_stats(training_data)

output_directory_path = '/shared/3/projects/sermons-ir/lda_model'

topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet,
                                                              output_directory_path,
                                                              NUM_TOPICS,
                                                              training_data)

assert(len(topic_distributions) == len(training_data))

for i, t in enumerate(topic_keys):
    print(i, '\t', ' '.join(t[:10]))

for p, d in lmw.get_top_docs(training_data, topic_distributions, topic_index=0, n=3):
    print(round(p, 4), d)
    print()
