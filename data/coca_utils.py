## data util file for COCA
import os
from tqdm import tqdm
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import pandas as pd
import re

SPLIT_CONST = '@ @ @ @ @ @ @ @ @ @'
YEAR_RANGE = [str(yr) for yr in range(1990, 2018)]
GENRES = ['academic_rpe', 'fiction_awq', 'magazine_qch', 'newspaper_lsp', 'spoken_kde']

# preprocess text data -- right now just splits by '@' sequences
def prep_COCA(txt_file):
    return txt_file.split(SPLIT_CONST)


# loads COCA from the raw directory
def load_COCA_from_raw(coca_root, just_one=False, join_text=False):
    year_genre_text_dict = {y: {g: [] for g in GENRES} for y in YEAR_RANGE}
    file_count = sum(len(files) for _, _, files in os.walk(coca_root))
    wrote = False

    dfs = []
    for subdir, dirs, files in tqdm(os.walk(coca_root), total=file_count):
        genre = subdir.replace('/data/laviniad/COCA/raw/text_', '')

        for file in files:
            file_path = os.path.join(subdir, file)
            #print(file_path)
            if file_path.endswith('.txt') and ((not just_one) or (not wrote)):
                year = re.findall(r'\d+', file_path)[0]
                with open(file_path) as f:
                    file_text = f.read()

                    if join_text:
                        file_text = '\n'.join(prep_COCA(file_text))
                        dfs.append({'year': year, 'genre': genre, 'text': file_text})
                    else:
                        for f in prep_COCA(file_text):
                            dfs.append({'year': year, 'genre': genre, 'text': f})
    
    print(f"Loaded {len(dfs)} rows...")
    return pd.DataFrame(dfs, columns=['year', 'genre', 'text'])


# turns year -> genre -> text dict into one dataframe
def output_comprehensive_df(coca_dicts, output_path=None):
    full_df = []

    for year in YEAR_RANGE:
        for g in GENRES:
            full_df.append({'year': year, 'genre': g, 'text': coca_dicts[year][g]})

    full_df = pd.DataFrame(full_df)
    return full_df
