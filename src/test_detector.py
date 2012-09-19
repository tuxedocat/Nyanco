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
try:
    from lsa_test.irstlm import *
except:
    pass
#=====================================================================================================

class TestLMDetector:
    def setUp(self):
        self.corpuspath = "../sandbox/fce_corpus/fce_processed.pickle"
        self.detector = LM_Detector(self.corpuspath)
        self.testlm_path = "../sandbox/irstlm_sample/testlm.gz"
        self.paslm_path = "../sandbox/pas/test_tsvout_huge_PAS.pickle"
        self.reportpath = "../sandbox/report.log"

    @attr("makecase")
    def test_makecase(self):
        self.detector.make_cases()
        print pformat(self.detector.testcases)
        raise Exception

    @attr("preLM")
    def test_preLM(self):
        lm = initLM(5, self.testlm_path)
        sc1 = getSentenceScore(lm, "the cat is black")
        logging.debug(pformat(sc1))
        raise Exception

    @attr("LM")
    def test_LM(self):
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_IRSTLM=self.testlm_path)
        self.detector.LM_count()
        print pformat(self.detector.testcases)
        raise Exception

    @attr("pasLM")
    def test_pasLM(self):
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_PASLM=self.paslm_path)
        self.detector.PASLM_count()
        print pformat(self.detector.testcases)
        raise Exception

    @attr("pasLM_ukwac")
    def test_pasLM_uk(self):
        paslm_path = "/work/yu-s/cl/nldata/PAS/ukwac_PAS.pickle"
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_PASLM=paslm_path)
        self.detector.PASLM_count()
        print pformat(self.detector.testcases)
        raise Exception      

    @attr("pasLM_eg")
    def test_pasLM_eg(self):
        paslm_path = "/work/yu-s/cl/nldata/PAS/eg_afp_apw_PAS.pickle"
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_PASLM=paslm_path)
        self.detector.PASLM_count()
        print pformat(self.detector.testcases)
        raise Exception        

    @attr("pasLM_eg+")
    def test_pasLM_eguk(self):
        paslm_path = "/work/yu-s/cl/nldata/PAS/eg_plus_ukwac_PAS.pickle"
        self.detector.make_cases()
        self.detector.read_LM_and_PASLM(path_PASLM=paslm_path)
        self.detector.PASLM_count()
        print pformat(self.detector.testcases)
        logging.debug(pformat(("Valid", self.detector.validnum_plm)))
        logging.debug(pformat(("Invalid", self.detector.invalidnum_plm)))
        raise Exception

    @attr("detect_small")
    def test_detect(self):
        detectmain(corpuspath=self.corpuspath, lmpath=self.testlm_path, paslmpath=self.paslm_path, reportout=self.reportpath)
        raise Exception