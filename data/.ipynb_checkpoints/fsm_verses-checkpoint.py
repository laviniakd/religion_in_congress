from rapidfuzz import fuzz
from data.bible_utils import comp_bible_helper
from data.data_utils import verse_range_to_individual_list
import spacy
import re
from spacy.pipeline import Sentencizer
nlp = spacy.load("en_core_web_sm")
config = {"punct_chars": ['!', '.', '?', '...', ';', ':', '(', ')']}
nlp.add_pipe("sentencizer", config=config)

VERSE_REGEX = r'\b(?:\d\s\w+\s\d+:\d+-\d+|\d\s\w+\s\d+:\d+|\w+\s\d+:\d+-\d+|\w+\s\d+:\d+)\b'

df = comp_bible_helper()
cols = ['King James Bible', 'American Standard Version', 'Douay-Rheims Bible', 'Darby Bible Translation', 'English Revised Version', 'Webster Bible Translation', 'World English Bible','Young\'s Literal Translation','American King James Version','Weymouth New Testament']
df['citation'] = df['citation'].apply(lambda x: x.lower())
all_citations = set(df['citation'].values)

def get_verse_versions(target_verse):
    versions = []
    
    if target_verse in all_citations:
        citation_row = df[df['citation'] == target_verse]
        for c in cols:
            versions.append(citation_row[c].values[0])
    return versions
                                      

def find_matching_verse(sentence, verse_versions, threshold=90):
    matched_version, score = max([(verse, fuzz.partial_ratio(sentence, verse)) for verse in verse_versions], key=lambda x: x[1])
    return matched_version if score >= threshold else None


#def mask_multi_sentence_verses(sentences, bible_verses_set, mask='[VERSE QUOTE]', threshold=90):
#    result = []
#    current_verse = None

#    for sentence in sentences:
#        verse = find_matching_verse(sentence, bible_verses_set, threshold=threshold)

#        if verse:
#            if current_verse is None:
#                current_verse = verse
#            elif current_verse != verse:
#                result.append(mask)
#                current_verse = verse
#        elif current_verse:
#            current_verse += ' ' + sentence
#        else:
#            result.append(sentence)

#    if current_verse:
#        result.append(mask)

#    return result

# ignoring boolean zen for the sake of readability
def begins_with_citation(sentence):
    citation = re.match(VERSE_REGEX, sentence)
    if citation:
        return sentence.startswith(citation[0])
    return False


# const is a reliable signal in sermoncentral that a verse has just been quoted
# as is the sentence beginning with a citation
def mask_verses(sentences, bible_verses_set, mask='[VERSE QUOTE]', threshold=90, const='read more »'):
    return [mask if (find_matching_verse(sentence, bible_verses_set, threshold=threshold) or (const in sentence) or begins_with_citation(sentence)) else sentence for sentence in sentences]
                                      

def mask_verse_quotes_in_sermon(input_sermon, target_verse, sentence_tokenizer=None, mask='[VERSE QUOTE]', threshold=90, mask_citations=True):
    target_list = verse_range_to_individual_list(target_verse) # convert ranges to lists of individual verses
    verse_versions = [get_verse_versions(target) for target in target_list] # get all versions of each verse/convert to text instead of citation
    if sentence_tokenizer is None:
        sentences = input_sermon.split('. ') # very janky
    else:
        sentences = sentence_tokenizer(input_sermon)
    
    if verse_versions != []:
        verse_versions = [e for s in verse_versions for e in s] # flatten
        bible_verses_set = set(verse_versions) # remove duplicate text (i.e., if two versions have same text for a given verse)

        if len(bible_verses_set) != 0:
            sentences = mask_verses(sentences, bible_verses_set, mask=mask, threshold=threshold)

    if mask_citations:
        sentences = [re.sub(VERSE_REGEX, '[CITATION]', s) for s in sentences]
    return ' '.join(sentences)