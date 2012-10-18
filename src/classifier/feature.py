#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/feature.py
Created on 18 Oct. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from numpy import array
from pattern.text import en
from nltk.corpus import wordnet as wn
try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *


class FeatureExtractorBase(object):
    def __init__(self):
        pass

    
    def preprocess(self):
        raise NotImplementedError


    @classmethod
    def ngram(self, wordlist=[]):
        raise NotImplementedError

    @classmethod
    def read_wordnet_tag(self):
        raise NotImplementedError

    @classmethod
    def convert_to_scipy(self, casedict={}):
        raise NotImplementedError


    def save_to_file(self):
        raise NotImplementedError


class SimpleFeatureExtractor(FeatureExtractorBase):
    """
    Extract features for given case, 
    this one has Surface/POS ngram, dependency, Semantic tags, and SRL related features 
    introduced by Liu et.al 2010

    More interesting features such as sentence wise topic models will be implemented in next update
    """
    def __init__(self):
        pass

