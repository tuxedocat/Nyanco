# ! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/detector.py
Created on 18 Oct 2012
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
import tool.altword_generator as altgen
from sklearn import cross_validation

class ClassifierExample(object):
    pass


def extract_training_examples(ukwac_prefix = "", verbset_path = "", max_num = ""):
    """
    Extract training examples from given (parsed and separated) ukWaC corpus and given verbset
    """
    import glob
    filelist = glob.glob(ukwac_prefix+"*.parsed")
    for cfname in filelist:
        corpus = open(cfname, 'r').read().split("\n\n")
        corpus = [s.split("\n") for s in corpus]


    return None