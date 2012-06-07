# /usr/env/bin python
# -*- coding: utf-8 -*-

import logging
import gensim
from gensim import corpora

class Corpus(object):
    def __init__(self, filename):
        corpusfile = open(filename, 'r')

    def __iter__(self):
        for line in corpusfile:
            yield dictionary.doc2bow(line.lower().split())
