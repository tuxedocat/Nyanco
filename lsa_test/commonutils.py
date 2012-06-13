#! /usr/bin/env python
# coding: utf-8


#------------------------------------------------------------------------------
# ngram extractor from string                                                  
#------------------------------------------------------------------------------
from nltk import sent_tokenize

def ngram(line, center="", window=5, test=False):
    '''
    function ngram
    @return
    list of list of ngrams
    [[-2,-1,0,1,2,3], [...]]
    '''
    sents = sent_tokenize(line)
    ngrams = [_retrieve_ngram(sent, center, window, test) for sent in sents]
    return ngrams


def _retrieve_ngram(sent, center, window, test):
    words = sent.split()
    try:
        c_index = words.index(center)
        if test is True:
            ngram = [word for index, word in enumerate(words) 
                        if index != c_index and index >= c_index - window and index <= c_index + window]
        else:
            ngram = [word for index, word in enumerate(words) 
                        if index >= c_index - window and index <= c_index + window]
        if ngram:
            return ngram
    except ValueError:
        return []
#------------------------------------------------------------------------------
