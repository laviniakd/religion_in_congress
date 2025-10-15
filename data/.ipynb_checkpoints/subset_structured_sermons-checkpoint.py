# just subset to rows with references + longer than MIN_LENGTH whitespace tokens

import pandas as pd

MIN_LENGTH = 25
BIBLE_PATH = '/home/laviniad/projects/religion_in_congress/data/bibles/AmericanKJVBible.txt'
FUZZY_CITATION_DICT_PATH = '/data/laviniad/sermons-ir/fuzzy_citation_dict.json'
OUT = '/data/laviniad/sermons-ir/sermoncentral/sermons_clean_text_have_refs.csv'
sermon_path = '/data/laviniad/sermons-ir/sermoncentral/sermons_clean_text.csv'

sermon_df = pd.read_csv(sermon_path, header=None)
    
sermon_df.columns = ['index', 'link', 'denomination',
                        'author', 'churchName', 'churchAddress',
                        'unknown1', 'unknown2', 'unknown3',
                        'date', 'title', 'versesList',
                        'topicsList', 'ratingsNumber',
                        'rating', 'text']

limited = sermon_df[sermon_df['versesList'].apply(lambda x: (x != '') and (x != []))] # depending on how it loads
limited = limited[limited['text'].apply(lambda x: isinstance(x, str) and (len(x.split()) >= MIN_LENGTH))]
limited.to_csv(OUT, header=None)
print(limited.head())
print("Done!")
