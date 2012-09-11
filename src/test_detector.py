#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/test_detector.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import logging
from pprint import pformat
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from corpusreader2 import *
import preprocessor2
import detector
from nose.plugins.attrib import attr
#=====================================================================================================

class TestLMDetector(object):
    def SetUp(self):
        self.corpuspath = "../sandbox/fce_corpus/fce_processed.pickle"
        self.detector = detector.LM_Detector(self.corpuspath)

    @attr("makecase")
    def test_makecase(self):
        self.detector.make_cases()

