# this is based on the 2016 ACL paper "Non-Literal Text Reuse in Historical Texts: An Approach to Identify Reuse Transformations and its Application to Bible Reuse"
# by Maria Moritz, Andreas Wiederhold, Barbara Pavlek, Yuri Bizzoni, and Marco Büchler
# https://aclanthology.org/D16-1190/

import re
import json
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 


# auxiliary functions described in paper
def checkm(morph1, morph2, pos_dict):
    if morph(morph1, pos_dict) == morph(morph2, pos_dict):
        return f"checkm:{morph1},{morph2}"
    else:
        return False
    

def morph(word, pos_dict):
    return pos_dict[word]


def lower(t,b):
    return f"LOWER:{t},{b}"


def upper(t,b):
    return f"UPPER:{t},{b}"


def lem(t,b):
    return f"LEM:{t},{b}"


def repl_syn(t,b):
    return f"REPL_SYN:{t},{b}"


def repl_hyper(t,b):
    return f"REPL_HYPER:{t},{b}"


def repl_hypo(t,b):
    return f"REPL_HYPO:{t},{b}"


# same hypernym
def repl_co_hypo(t,b):
    return f"REPL_CO_HYPO:{t},{b}"


def NOP(t, b):
    return f"NOP:{t},{b}"


# same case/POS
def NOPmorph(t, b):
    return f"NOPmorph:{t},{b}"


# same cognate, POS changed
def repl_pos(t,b):
    return f"REPL_POS:{t},{b}"


def repl_case(t,b):
    return f"REPL_CASE:{t},{b}"


def no_rel_found(t,b):
    return f"NO_REL:{t},{b}"


# L: dict of word-lemma pairs obtained from resources
# S: dict of synsets
# T: list of words of reuse instance
# B: list of words of in each Bible verse
def capture_rephrase_ops(text1, text2, L, S, T, B, OP):
    op_list = []
    tmp_op = None
    for t in T:
        for b in B:
            if t == b:
                op_list.append(NOP(t,b), checkm(morph(t), morph(b)))
            elif t.lower() == b:
                op_list.append(lower(t,b), checkm(morph(t), morph(b)))
            elif t.upper() == b:
                op_list.append(upper(t,b), checkm(morph(t), morph(b)))
            elif t in S.keys() and b in S.keys():
                s1 = S[t]
                s2 = S[b]
                if s1 == s2:
                    op_list.append(repl_syn(t,b))
                # check if t is hypernym of b
                elif t in s2.hypernyms():
                    op_list.append(repl_hyper(t,b))
                elif t in s2.hyponyms():
                    op_list.append(repl_hypo(t,b))
                elif s1.hypernyms() == s2.hypernyms():
                    op_list.append(repl_co_hypo(t,b))
            else:
                tmp_op = no_rel_found(t,b)
        
    if tmp_op:
        op_list.append(tmp_op)
    else:
        op_list.append(f"unknown:{t}")


    return op_list


def get_rephrases_for_list_of_texts(text_list_1, text_list_2):
    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("spacy_wordnet", after='tagger')

    rephrases = []
    for text1, text2 in zip(text_list_1, text_list_2):
        doc1 = nlp(text1)
        doc2 = nlp(text2)

        L = {t.text_: t.lemma_ for t in doc1}
        S = {t.text_: t._.wordnet.synsets() for t in doc1}
        # get pos tags
        T = {t.text: t.pos_ for t in doc1}
        B = {t.text: t.pos_ for t in doc2}
        OP = {} # should be "list of sets containing up to 3 parametrized operations"
        rephrases.append(capture_rephrase_ops(text1, text2, L, S, T, B, OP))
    return rephrases
