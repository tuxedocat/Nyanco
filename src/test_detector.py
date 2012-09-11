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
from detector import *
from nose.plugins.attrib import attr
#=====================================================================================================

class TestLMDetector:
    def setUp(self):
        self.corpuspath = "../sandbox/fce_corpus/fce_processed.pickle"
        self.detector = LM_Detector(self.corpuspath)

    @attr("makecase")
    def test_makecase(self):
        self.detector.make_cases()
        print pformat(self.detector.testcases)
        raise Exception
