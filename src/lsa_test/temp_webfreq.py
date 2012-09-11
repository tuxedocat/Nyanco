#! /usr/bin/env python
# coding: utf-8
'''
webfreq.py

count frequency of given queries on Web1T (Google web ngram)
this actually works with IRSTLM wrapper
'''

import os, sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
from collections import defaultdict


class Webfrequency(object):
    '''
    this class is for estimating frequency of given list of queries 
    on the Web1T corpus using IRSTLM
    '''
    def __init__(self, lm_path, lm_N):
        from irstlm import *
        logging.info('Started loading LM... this may take hours')
        self.LM = initLM(lm_N, lm_path)
        logging.info('Finished loading LM, now you can go...')
    

    def readqueries(self,queries):
        self.queries = queries
    

    


