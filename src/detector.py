#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/detector.py
Created on 9 Sep. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

import os
import nltk
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from datetime import datetime
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)


class DetectorBase(object):
    def __init__(self, corpusdictpath):
        if os.path.exists(corpusdictpath):
            with open(corpusdictpath, "rb") as f:
                corpusdict = pickle.load(f)
                self.corpus = corpusdict
        else:
            raise IOError

    def make_cases(self):
        self.testcases = defaultdict(dict)
        for docname, doc in self.corpus.iteritems():
            try:
                gold_tags = doc["gold_tags"]
                test_tags = doc["RVtest_tags"]
                gold_text = doc["gold_text"]
                test_text = doc["RVtest_text"]
                gold_pas = doc["gold_PAS"]
                test_pas = doc["RVtest_PAS"]
                errorpositions = doc["errorpositions"]
            except KeyError as e:
                pass
        raise NotImplementedError

    def detect(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError



class LM_Detector(DetectorBase):
    def read_LM_and_PASLM(self, path):
        pass

    def detect(self):
        pass
