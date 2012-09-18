#! /usr/bin/env python
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
        self.handler = corpushandler.CorpusHandler(self.corpuspath)

    @attr("full","filter")
    def test_filter(self):
        self.handler.filter_checkpoints()
        # print pformat(self.handler.processedcorpus.items()[0:100])
        raise Exception

    @attr("full", "parse")
    def test_parse_pre(self):
        self.handler.filter_checkpoints()
        self.handler.parse_eachsent_pre()
        raise Exception

    @attr("full", "parse_read")
    def test_parse_read(self):
        self.handler.filter_checkpoints()
        self.handler.parse_eachsent_read()
        logging.debug(pformat(self.handler.processedcorpus.items()[0:10]))
        raise Exception


    @attr("full", "filter_others")
    def test_filter2(self):
        self.handler.filter_others()
        print pformat(self.handler.processedcorpus.items()[0:10])
        # self.handler.parse_eachsent_pre_others()
        raise Exception
