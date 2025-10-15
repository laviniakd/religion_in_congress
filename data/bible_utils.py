import pandas as pd
from tqdm import tqdm


# want to take in bible df and optional list of verses (in citation format) to avoid, then return text of verse
def build_candidates(bible_df, subset=None):
    if subset is not None:
        new_df = bible_df[bible_df['citation'].apply(lambda x: x in subset)]
        return new_df['text']

    return bible_df['text'].apply(str).tolist()


# deal with variant formatting: 'Revelation 1:4' or '1 Thessalonians 2:5'
def verse_extractor(v):
    if v.startswith('\ufeff'):
        v = 'Genesis 1:1'  # bug :(

    verse_split = v.split()
    if verse_split[0].isnumeric(): # int book format
        book = ' '.join(verse_split[:2])
        cv = verse_split[2].split(':')
        chapter = int(cv[0])
        verse = int(cv[1])
        text = ' '.join(verse_split[3:])
    elif verse_split[0] != 'Song': # normal format
        book = verse_split[0]
        cv = verse_split[1].split(':')
        chapter = int(cv[0])
        verse = int(cv[1])
        text = ' '.join(verse_split[2:])
    else: # song of solomon is excepted bc multiple (whitespace) tokens
        book = 'Song of Solomon'
        cv = verse_split[3].split(':')
        chapter = int(cv[0])
        verse = int(cv[1])
        text = ' '.join(verse_split[4:])

    citation = book + ' ' + str(chapter) + ':' + str(verse)

    return book, chapter, verse, text, citation

def reduced_verse_extractor(v):
    # just processes citation
    vsplit = v.split()
    cv = vsplit[-1]
    book = ' '.join(vsplit[:-1])
    chapter, verse = cv.split(':')
    #print(book, chapter, verse)
    
    return str(book), str(chapter), str(verse), str(v)


def bible_helper(INPUT_PATH):
    with open(INPUT_PATH, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    bible_df = []
    for l in lines:
        b, c, v, t, cit = verse_extractor(l)
        bible_df.append({'book': b, 'chapter': c, 'verse': v, 'text': t, 'citation': cit})

    return pd.DataFrame(bible_df)


def comp_bible_helper(INPUT_PATH='/home/laviniad/projects/religion_in_congress/data/bibles/bibles.txt'):
    df = pd.read_csv(INPUT_PATH, sep='\t', encoding='latin-1')
    assert('Verse' in df.columns) # just a check
    #df[['book', 'chapter', 'verse', 'citation']] = df['Verse'].apply(lambda x: reduced_verse_extractor(x), result_type='expand')
    df[['book', 'chapter', 'verse', 'citation']] = df['Verse'].apply(lambda x: pd.Series(reduced_verse_extractor(x)))
    
    return df
    

