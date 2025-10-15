# sample random 1,000,000 from CR
import argparse

from src.references.retrieve_and_infer import load_embedding_model, load_bible_data_for_references, load_and_filter_cr, create_inference_df, get_embedded_verses, create_verse_loader, create_congress_loader, embed_cr_sentences_and_match
from data import congress_utils

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=1000000)
args = parser.parse_args()

SAMPLE = args.sample
MIN_SENTENCE_LENGTH = 5
PATH = "/data/corpora/congressional-record/"

BIBLE_VERSION = "King James Bible"
verse_df, limited_verse_to_citation, limited_citation_to_verse = load_bible_data_for_references(version=BIBLE_VERSION)
cr_df = congress_utils.load_full_df_from_raw(PATH)
cr_df = create_inference_df(MIN_SENTENCE_LENGTH, cr_df)
cr_df = cr_df.sample(args.sample)

# dump cr_df to file
cr_df.to_csv(f"/data/laviniad/sermons-ir/references/test/cr_df_{SAMPLE}.csv")

