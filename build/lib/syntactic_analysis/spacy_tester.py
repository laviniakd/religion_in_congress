# i want to make sure i'm working with the dependency parsing labels correctly...

import spacy
nlp = spacy.load("en_core_web_md")

text1 = "May God bless you." # simple case, with imperative
text2 = "We see God's mercy here today." # this is a possessive case
text3 = "God bless America." # simplest case -- God is root of sentence
text4 = "The other countries' God is our God." # possessive case with multiple possessors

token = "God"
nlp_text1 = nlp(text1)
nlp_text2 = nlp(text2)
nlp_text3 = nlp(text3)
nlp_text4 = nlp(text4)


def get_possession_data(token):
    possessed_by, possessed = None, None
    if token.dep_ == "poss":
        possessed = token.head.text
    
    # will assume for now that there is only one possessor...
    for child in token.children:
        if child.dep_ == "poss":
            possessed_by = child.text

    return possessed_by, possessed


for t in [nlp_text1, nlp_text2, nlp_text3, nlp_text4]:
    for token in t:
        if token.text == "God":
            print(get_possession_data(token))

            print(token.text, token.dep_, token.head.text, token.head.pos_,
                  [child for child in token.children], [child for child in token.ancestors])
            print("\n")
