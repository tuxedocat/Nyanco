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
        self.testlm_path = "../sandbox/irstlm_sample/testlm.gz"

    @attr("makecase")
    def test_makecase(self):
        self.detector.make_cases()
        print pformat(self.detector.testcases)
        raise Exception

    @attr("LM")
    def test_LM(self):
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_IRSTLM=self.testlm_path)
        self.detector.LM_count()
        print pformat(self.detector.testcases)
        raise Exception
