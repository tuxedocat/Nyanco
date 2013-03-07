# coding: utf-8
'''
Nyanco/src/test_corpushandler.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import logging
from pprint import pformat
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import corpushandler
from nose.plugins.attrib import attr
#=====================================================================================================

class TestCorpusHandler:
    def setUp(self):
        import os, sys
        import collections
        self.corpuspath = "../sandbox/fce_corpus/fce.pickle"
        self.handler = corpushandler.CorpusHandler(self.corpuspath, outputname="")

    @attr("filter")
    def test_filter(self):
        self.handler.filter_checkpoints()
        logging.debug(pformat(self.handler.processedcorpus.items()[-5:]))
        raise Exception

    @attr("pre-parse")
    def test_parse_pre(self):
        self.handler.filter_checkpoints()
        self.handler.parse_eachsent_pre()
        raise Exception

    @attr("parseread")
    def test_parse_read(self):
        self.handler.filter_checkpoints()
        self.handler.parse_eachsent_read()
        logging.debug(pformat(self.handler.processedcorpus.items()[0:10]))
        raise Exception


    @attr("full", "VB")
    def test_VB(self):
        self.handler = corpushandler.CorpusHandler(self.corpuspath, outputname="fce_VB.pickle3")
        self.handler.filter_others()
        # print pformat(self.handler.processedcorpus.items()[0:5])
        self.handler.onlineparse_others()
        logging.debug(pformat(self.handler.processedcorpus_others[0:5]))
        raise Exception

    @attr("full", "RV")
    def test_RV(self):
        self.handler = corpushandler.CorpusHandler(self.corpuspath, outputname="fce_RV.pickle3")
        self.handler.filter_checkpoints()
        # print pformat(self.handler.processedcorpus.items()[0:5])
        self.handler.onlineparse_RV()
        # logging.debug(pformat(self.handler.processedcorpus[0:5]))
        raise Exception
