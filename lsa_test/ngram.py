#! /usr/bin/env python
# coding: utf-8

from nltk import sent_tokenize

def ngram(line, center=None, window=5):
    sents = sent_tokenize(line)
    ngrams = []
    for sent in sents:
        ngram = _retrieve_ngram(sent, center, window)
        ngrams.append(ngram)
   

def _retrieve_ngram(sent, center=None, window=5):
    words = sent.split()
    c_index = words.index(center)
    ngram = [(index, word)for index, word in enumerate(words) 
                if index != c_index and index >= c_index - window]
    return ngram
