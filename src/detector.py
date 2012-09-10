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
from datetime import datetime
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)


class DetectorBase(object):
    def __init__(self, corpusdict):
        self.corpus = corpusdict

    def make_cases(self):
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
