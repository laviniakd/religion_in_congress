# import unittest # ?
# TODO: fill this out with data-related function tests
# Modeling should come later...
from data.data_utils import load_stop_words


def data_tests():
    def test_stop_words():
        STOP_WORDS = load_stop_words()
        
        some_stop_words = ['and', 'the', 'not']
        not_stop_words = ['language', 'word', 'politics']
        for s in some_stop_words:
            assert(s in STOP_WORDS)
        for s in not_stop_words:
            assert(s not in STOP_WORDS)

        # get name of func? idk python
        print("PASSED: test_stop_words")

    test_stop_words()


data_tests()
