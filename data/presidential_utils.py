from tqdm import tqdm
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from data.data_utils import count_files
import pandas as pd
import json
import os

YEAR_RANGE = [str(yr) for yr in range(1990, 2024)]
GENRES = ['inaugurals', 'sotu_addresses', 'spoken_addresses', 'written_messages', 'farewell', 'letters', 
        'news_conferences', 'sotu_messages', 'statements', 'written_statements']
actual_presidents = ['John Tyler', 'Calvin Coolidge', 'Martin van Buren', 'Andrew Johnson', 'Franklin D. Roosevelt', 'Theodore Roosevelt', 'Warren G. Harding', 'William McKinley', 'William Henry Harrison', 'Ronald Reagan', 'John F. Kennedy', 'Joseph R. Biden', 'Herbert Hoover', 'Abraham Lincoln', 'Thomas Jefferson', 'Benjamin Harrison', 'Franklin Pierce', 'James A. Garfield', 'Harry S. Truman', 'John Adams', 'James Monroe', 'Rutherford B. Hayes', 'Jimmy Carter', 'Lyndon B. Johnson', 'William Howard Taft', 'Barack Obama', 'Millard Fillmore', 'James K. Polk', 'Ulysses S. Grant', 'Richard Nixon', 'Grover Cleveland', 'Chester A. Arthur', 'James Buchanan', 'Zachary Taylor', 'William J. Clinton', 'Walter F. Mondale', 'George Bush', 'George W. Bush', 'Andrew Jackson', 'George Washington', 'James Madison', 'Donald J. Trump', 'Dwight D. Eisenhower', 'Woodrow Wilson', 'Gerald R. Ford', 'John Quincy Adams']
party_mapping = {
    'John Tyler': 'Whig/Independent',
    'Calvin Coolidge': 'Republican',
    'Martin van Buren': 'Democratic',
    'Andrew Johnson': 'Democratic',
    'Franklin D. Roosevelt': 'Democratic',
    'Theodore Roosevelt': 'Progressive/Republican',
    'Warren G. Harding': 'Republican',
    'William McKinley': 'Republican',
    'William Henry Harrison': 'Whig',
    'Ronald Reagan': 'Republican',
    'John F. Kennedy': 'Democratic',
    'Joseph R. Biden': 'Democratic',
    'Herbert Hoover': 'Republican',
    'Abraham Lincoln': 'Republican',
    'Thomas Jefferson': 'Democratic-Republican',
    'Benjamin Harrison': 'Republican',
    'Franklin Pierce': 'Democratic',
    'James A. Garfield': 'Republican',
    'Harry S. Truman': 'Democratic',
    'John Adams': 'Federalist',
    'James Monroe': 'Democratic-Republican',
    'Rutherford B. Hayes': 'Republican',
    'Jimmy Carter': 'Democratic',
    'Lyndon B. Johnson': 'Democratic',
    'William Howard Taft': 'Republican',
    'Barack Obama': 'Democratic',
    'Millard Fillmore': 'Whig',
    'James K. Polk': 'Democratic',
    'Ulysses S. Grant': 'Republican',
    'Richard Nixon': 'Republican',
    'Grover Cleveland': 'Democratic',
    'Chester A. Arthur': 'Republican',
    'James Buchanan': 'Democratic',
    'Zachary Taylor': 'Whig',
    'William J. Clinton': 'Democratic',
    'Walter F. Mondale': 'Democratic',
    'George Bush': 'Republican',
    'George W. Bush': 'Republican',
    'Andrew Jackson': 'Democratic',
    'George Washington': 'Unaffiliated',
    'James Madison': 'Democratic-Republican',
    'Donald J. Trump': 'Republican',
    'Dwight D. Eisenhower': 'Republican',
    'Woodrow Wilson': 'Democratic',
    'Gerald R. Ford': 'Republican',
    'John Quincy Adams': 'Democratic-Republican'
}
usual_root = '/data/laviniad/presidential/'
directory_to_ignore = [usual_root + e for e in ['', 'roberta_predictions', 'press_briefings', 'misc_remarks']]

# preprocess text data -- right now doesn't do anything
def prep_presidential(txt_file):
    return txt_file


def load_full_df_from_raw(presidential_root, truncate_length=-1, tokenizer=None, debug=False):
    file_count = count_files(presidential_root)
    full_df = []
    
    for subdir, dirs, files in tqdm(os.walk(presidential_root), total=file_count):
        genre = subdir.replace('/data/laviniad/presidential/', '')

        if subdir not in directory_to_ignore:
            for file in files: # should be just 1 for each genre directory
                if file.endswith('.jsonlist'):
                    year_to_text = {y: '' for y in YEAR_RANGE}
                    file_path = os.path.join(subdir, file)
                    #print(file_path)
                    with open(file_path) as f:
                        lines = f.readlines()
                        if debug:
                            lines = lines[:5]

                    for l in lines:
                        try:
                            obj = json.loads(l)
                            person = obj['person']
                            is_president = person in actual_presidents
                            if is_president:
                                party_name = party_mapping[person]
                            else:
                                party_name = 'NA'
                            results_dict = {
                                'year': obj['date'].split(', ')[1],
                                'speaker': person,
                                'text': '\n'.join(obj['text']), # is paragraph-tokenized in data
                                'genre': genre,
                                'party': party_name,
                                'title': obj['title'],
                                'is_president': is_president
                            }
                            if truncate_length > 0 and tokenizer is not None:
                                truncated_and_tokenized_text = tokenizer.tokenize(results_dict['text'])[:truncate_length]              
                                token_ids = tokenizer.convert_tokens_to_ids(truncated_and_tokenized_text)
                                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                                results_dict['trunc_and_tokenized_text'] = decoded_text
                                
                            full_df.append(results_dict)
                        except json.decoder.JSONDecodeError:
                            print("couldn't decode from file " + file_path)

    return pd.DataFrame(full_df)


# loads presidential from the raw directory
def load_presidential_from_raw(presidential_root, debug=False):
    
    file_count = sum(len(files) for _, _, files in os.walk(presidential_root))
    year_genre_text_dict = {y: {g: [] for g in GENRES} for y in YEAR_RANGE}
    for subdir, dirs, files in tqdm(os.walk(presidential_root), total=file_count):
        genre = subdir.replace('/data/laviniad/presidential/', '')

        if subdir not in directory_to_ignore:
            for file in files: # should be just 1 for each genre directory
                if file.endswith('.jsonlist'):
                    year_to_text = {y: '' for y in YEAR_RANGE}
                    file_path = os.path.join(subdir, file)
                    #print(file_path)
                    with open(file_path) as f:
                        lines = f.readlines()
                        if debug:
                            lines = lines[:5]

                    for l in lines:
                        try:
                            obj = json.loads(l)
                            year = obj['date'].split(', ')[1]
                            person = obj['person']
                            if year in YEAR_RANGE and person in actual_presidents:
                                year_to_text[year] += '\n' + prep_presidential(' '.join(obj['text']))
                        except json.decoder.JSONDecodeError:
                            print("couldn't decode from file " + file_path)

                    for y in year_to_text.keys():
                        year_genre_text_dict[y][genre] = word_tokenize(year_to_text[y])

    return year_genre_text_dict


# turns year -> genre -> text dict into one dataframe
# no longer retaining word tokenizing!
def output_comprehensive_df(presidential_dicts, output_path=None):
    full_df = []
    print("Note: this method used to return a DF where the sermon text was word-tokenized. It does not do this anymore; each sermon is one string.")

    for year in YEAR_RANGE:
        for g in GENRES:
            full_df.append({'year': year, 'genre': g, 'text': ' '.join(presidential_dicts[year][g])})

    full_df = pd.DataFrame(full_df)
    if output_path is not None:
        full_df.to_csv(output_path)
        
    return full_df

