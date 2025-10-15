import json
import pandas as pd
import re
import os
import little_mallet_wrapper as lmw
from rapidfuzz import fuzz
from rapidfuzz.process import extractOne
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from data.bible_utils import bible_helper


LOAD_SPACY = False

nlp = None

from collections import Counter
import nltk


# see january notes for source
pattern = r'''(?x)          # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
    '''
tokenizer = RegexpTokenizer(pattern)


# keyword info
project_path = '/home/laviniad/projects/religion_in_congress/src/multi-feature-use/'
keyword_strs = project_path + 'keywords_from_coca.txt', project_path + 'keywords_from_congress.txt'

def get_keywords(keyword_path):
    with open(keyword_path) as f:
        keyword_set = [l.strip() for l in f.readlines()]
    return keyword_set

keywords_coca = get_keywords(keyword_strs[0])
keywords_congress = get_keywords(keyword_strs[1])

def load_jesus_synonyms():
    TOK = ['Jesus']
    wnl = WordNetLemmatizer()
    syn_jesus = wn.synsets(wnl.lemmatize("Jesus"), pos=wn.NOUN)
    actual = syn_jesus[0]
    jesus_list = list(set([l.name().replace('_', ' ') for l in actual.lemmas()]))
    #print(f"Synonym set: {jesus_list}")
    return jesus_list


# also take hyponyms
def load_god_synonyms():
    TOK = ['God']
    wnl = WordNetLemmatizer()
    syn = wn.synsets(wnl.lemmatize(TOK[0]), pos=wn.NOUN)
    actual = syn[0]
    hyponyms = actual.instance_hyponyms()[:-1] # last tends to be unreliable
    lemmas_of_hyponyms = [s.lemmas() for s in hyponyms]
    hyponyms = [str(s.name()) for instance in lemmas_of_hyponyms for s in instance]
    TOK += hyponyms
    TOK += [str(s.name()).capitalize() for s in actual.lemmas()]
    TOK = list(set([s.replace('_', ' ') for s in TOK if s != 'Allah']))
    temp = []
    for t in TOK:
        if t != 'Creator' and t != 'Maker' and t!= 'Lord':
            temp.append(t.lower())
    TOK += temp
    TOK = list(set(TOK))
    #print(f"Synonym set: {TOK}")
    return TOK


GOD_TOK = load_god_synonyms()
JESUS_TOK = load_jesus_synonyms()

def get_full_keywords():
    fk = list(set(keywords_congress).union(set([t.capitalize() for t in GOD_TOK + JESUS_TOK])))
    return fk

full_keywords = get_full_keywords()


def get_lexical_overlap(speech):
    count_dict = {t: [] for t in full_keywords}
    speech = tokenizer.tokenize(speech)
    
    if len(speech) > 0:
        count = 0
        for t in full_keywords:
            inst = speech.count(t)
            count += inst
            count_dict[t] = inst

        return count / len(speech), count_dict # roughly normalize
    return 0, count_dict


def get_simple_overlap(speech):
    norm_count, _ = get_lexical_overlap(speech)
    return norm_count

def get_lexical_overlap_unique(speech):
    speech = tokenizer.tokenize(speech)
    
    if len(speech) > 0:
        count = 0
        for t in full_keywords:
            if t in speech:
                count += 1

        return count / len(speech) # roughly normalize
    return 0

def get_relig_boolean(speech):
    speech = tokenizer.tokenize(speech)
    
    if len(speech) > 0:
        count = 0
        for t in full_keywords:
            if t in speech:
                return True

    return False


def load_spacy_sentencizer(punct_chars=None):
    import spacy
    from spacy.pipeline import Sentencizer
    nlp = spacy.load("en_core_web_sm")
    if not punct_chars:
        config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
    else:
        assert(isinstance(punct_chars, list))
        config = {"punct_chars": punct_chars}
    nlp.add_pipe("sentencizer", config=config)
    return nlp

    
if LOAD_SPACY:
    nlp = load_spacy_sentencizer()


VERSE_REGEX = r'\b(?:\d\s\w+\s\d+:\d+-\d+|\d\s\w+\s\d+:\d+|\w+\s\d+:\d+-\d+|\w+\s\d+:\d+)\b'

QUOTE_PATTERN = re.compile(r'\"[^\"]*\"')  # only matches text within odd-even quote number pairs
## NOTE: this means that an errant quote could screw things up.
QUOTE_STR = '[QUOTE]'


# get list of stop words from reference file
def load_stop_words():
    FILE = '/home/laviniad/projects/religion_in_congress/data/stop.txt'
    with open(FILE, 'r') as f:
        stop_words = f.readlines()
        stop_words = [s.split('|')[0].strip() for s in stop_words] # format of file: stop words not preceded by 

    return stop_words
        

# get dictionaries mapping various realizations of verse citations in the sermon dataset to their actual citations
# and mapping verse citations to their KJV text
def get_verse_dicts(fcd_path, bible_path):
    with open(fcd_path) as f:
        fuzzy_citation_dict = json.load(f)
    bible_df = bible_helper(bible_path)
    verse_text_dict = {(str(r.book) + ' ' + str(r.chapter) + ':' + str(r.verse)): r.text for r in bible_df.itertuples()}
    verse_text_dict = {k.lower(): v for k, v in
                       verse_text_dict.items()}  # fuzzy citation dict outputs are lower case right now

    return verse_text_dict, fuzzy_citation_dict


# limit number of fsm computations needed by using only recognized verses
def mask_quotes_using_fsm(row, citation_to_verse_dict, fuzzy_citation_dict,
                          fuzz_threshold=80, print_debug=False, length_minimum=8,
                          calc_all_scores=False):

    masked_verse_list = []
    fuzz_scores = []
    original_text_list = []
    sermon_string = row['text']
    recognized_verses = row['recognized_verses']

    try:
        verses = [citation_to_verse_dict[fuzzy_citation_dict[v]] for v in recognized_verses]
        in_fuzz_dict_count = len(verses)
        in_cv_dict_count = len(verses)
    except KeyError:
        row['corrected_text'] = ''
        row['masked_text'] = ''
        row['masked_verses'] = []
        row['fuzz_scores'] = []
        row['original_text_list'] = []
        return row

    if print_debug:
        print("in fuzz dict", str(in_fuzz_dict_count))
        print("in cv dict", str(in_cv_dict_count))
        print("verse_length: " + str(len(verses)))

    new_sermon = ''
    new_sermon_masked = ''

    assert (nlp is not None)
    docs = nlp(sermon_string, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    count = 0
    for temp in docs.sents:
        s = temp.text.strip()

        choice_v = None
        if len(s.split()) >= length_minimum:
            choice_v = extractOne(s, verses)

        if choice_v is not None and choice_v[1] > fuzz_threshold:
            new_sermon += ' ' + choice_v[0]  # REPLACES OLD STR WITH "CANONICAL" VERSION OF VERSE
            new_sermon_masked += ' [QUOTE].'
            original_text_list.append(s)
            masked_verse_list.append(choice_v[0])
            fuzz_scores.append(choice_v[1])
            count += 1
        else:
            new_sermon += ' ' + s
            new_sermon_masked += ' ' + s

            if calc_all_scores and choice_v is not None:
                fuzz_scores.append(choice_v[1])

    row['corrected_text'] = new_sermon
    row['masked_text'] = new_sermon_masked
    row['masked_verses'] = masked_verse_list
    row['fuzz_scores'] = fuzz_scores
    row['original_text_list'] = original_text_list

    return row


def mask_quotes_using_quotation_marks(sermon_string, DEBUG=False):
    if DEBUG:
        print("MASKING")
    last = ""

    idx = 0
    while last != sermon_string:
        last = sermon_string
        sermon_string = QUOTE_PATTERN.sub(QUOTE_STR, sermon_string)

        idx += 1

    return sermon_string


# takes in verse range string (e.g., 'John 3:16-18') and outputs list of verses contained 
# (e.g., ['John 3:16', 'John 3:17', 'John 3:18'])
def verse_range_to_individual_list(v):
    if '-' not in v:
        return [v]
    else:
        prefix, verse_range = v.split(':')
        start, end = verse_range.split('-')
        range_nums = range(int(start), int(end) + 1)

        v_list = [prefix + ':' + str(n) for n in range_nums]

    return v_list


def return_verses_in_passage(passage, split_ranges=False):
    assert (isinstance(passage, str))
    matches = re.findall(VERSE_REGEX, passage)
    if split_ranges:
        temp = []
        for m in matches:
            if '-' in m:
                temp = temp + verse_range_to_individual_list(m)
            else:
                temp.append(m)
        return temp
    else:
        return matches


def load_sermons(SERMON_PATH="/data/laviniad/sermons-ir/sermoncentral/sermons_clean_text.csv",
                 SAMPLE=False, CLEAN_MORE=True):
    sermon_df = pd.read_csv(SERMON_PATH, header=None)

    sermon_df.columns = ['index', 'link', 'denomination',
                         'author', 'churchName', 'churchAddress',
                         'unknown1', 'unknown2', 'unknown3',
                         'date', 'title', 'versesList',
                         'topicsList', 'ratingsNumber',
                         'rating', 'text']

    sermon_df.set_index('index', inplace=True)
    if SAMPLE:
        assert (isinstance(SAMPLE, int))
        sermon_df.sample(SAMPLE, inplace=True)

    # regex = re.compile(r'(\n|\\n)')
    if CLEAN_MORE:
        sermon_df['text'] = sermon_df['text'].apply(str)
        sermon_df['text'] = sermon_df['text'].apply(mask_quotes_using_quotation_marks)
        #sermon_df['text'] = sermon_df['text'].apply(lambda x: x.replace('\n', ' '))
    return sermon_df


def recognize_verse(t):
    pass


# adaptation from below
def get_verses_from_sermon_without_context(sent_toked_sermon, fuzzy_citation_dict):
    sermon_quals = []
    
    for i, s in enumerate(sent_toked_sermon):
        match = re.match(VERSE_REGEX, s)
        # need to know what match returns...
    
        if match:
            sent_idx = i
            verse = match.group(0)
            if verse in fuzzy_citation_dict.keys():
                verse = fuzzy_citation_dict[verse]
            else:
                verse = "UNKNOWN"
  
            # result_dict = {'verse': verse, 'sent_idx': sent_idx}
            # print(result_dict)
            sermon_quals.append(verse)

    return sermon_quals


def get_verses_from_sermon(sent_toked_sermon, fuzzy_citation_dict, N, M):
    sermon_quals = []
    
    for i, s in enumerate(sent_toked_sermon):
        match = re.match(VERSE_REGEX, s)
        # need to know what match returns...
    
        if match:
            verse_pre = sent_toked_sermon[i - N:i]
            verse_post = sent_toked_sermon[i + 1:i + M]
            masked_verse = re.sub(VERSE_REGEX, '[VERSE]', s)
            sent_idx = i
            verse = match.group(0)
            if verse in fuzzy_citation_dict.keys():
                verse = fuzzy_citation_dict[verse]
            else:
                verse = "UNKNOWN"
    
            context = ' '.join(verse_pre + [masked_verse] + verse_post)
            result_dict = {'context': context, 'verse': verse, 'sent_idx': sent_idx}
            # print(result_dict)
            sermon_quals.append(result_dict)

    return sermon_quals


# produces a bunch of context, verse, index_of_verse_sentence dfs
def split_by_verses(sermon_text, N, M, fuzzy_citation_dict, match_verse_text=False):
    try:
        assert (nlp is not None)
        docs = nlp(str(sermon_text), disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

        sent_toked_sermon = [t.text for t in docs.sents]
    except TypeError:
        print("TYPE ERROR")
        print(sermon_text)
        return None

    sermon_quals = get_verses_from_sermon(sent_toked_sermon, fuzzy_citation_dict, N, M)

    #print("Number of verse objects: " + str(len(sermon_quals)))
    sermon_quals = pd.DataFrame(sermon_quals)
    if sermon_quals.empty:
        return None

    return sermon_quals


# NOTE: for now, the "commentary" is the N sentences above and M below
# question: include other verses in commentary? you know what, why not
# also_match_verses controls whether the program also tries to string match for Bible verse text,
# as opposed to just verse citations a la "Hebrews 1:2"
def chunk_sermons(sermon_df, fuzzy_citation_dict, N=2, M=2, also_match_verses=False):
    SERMON_NUMBER = len(sermon_df.index)

    def chunk(sermon_df_row, id=None):
        sermon_text = sermon_df_row.text
        # assert(isinstance(sermon_text, str)) -- this is handled in split_by_verses
        # print(sermon_text)

        sermon_qual_df = split_by_verses(sermon_text, N, M, fuzzy_citation_dict, match_verse_text=also_match_verses)

        if sermon_qual_df is None:
            return None

        # sermon_qual_df['text'] = sermon_text
        if id:
            sermon_qual_df['id'] = id
        else:
            sermon_qual_df['id'] = sermon_df_row.index

        return sermon_qual_df

    chunked_dfs = []
    for i, r in tqdm(enumerate(sermon_df.itertuples()), total=SERMON_NUMBER):
        temp = chunk(r, id=i)  # takes rows and returns dfs?
        if temp is not None:
            chunked_dfs.append(temp)

    # index conflict?
    chunked_df = pd.concat(chunked_dfs).reset_index()

    # various debugging-y things
    SERMON_HAS_VERSE_CITATION = len(list(set(chunked_df['id'])))  # number of unique sermons represented in chunked df
    PERCENT = (float(SERMON_HAS_VERSE_CITATION) / float(SERMON_NUMBER)) * 100

    print("Number of sermons: " + str(SERMON_NUMBER))
    print("Number of sermons with captured verse citations: " + str(SERMON_HAS_VERSE_CITATION))
    print("This means " + str(round(PERCENT, 2)) + "% of sermons have captured verses in them.")
    # done

    return chunked_df


def count_files(directory):
    file_count = 0

    for root, dirs, files in os.walk(directory):
        file_count += len(files)

    return file_count
