# coding: utf-8
'''
Nyanco/src/test_corpusreader2
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
from nose.plugins.attrib import attr
#=====================================================================================================

class TestCorpusReader2:
    def setUp(self):
        import os, sys
        import glob
        import collections
        self.corpus_dir = '../sandbox/fce'
        self.working_dir = '../sandbox/fce'
        self.output_dir = '../sandbox/fce'

    def test_filelist(self):
        obtained = make_filelist(path=self.corpus_dir, prefix="doc", filetype="xml")
        expected = ['../sandbox/fce/doc100.xml',
                     '../sandbox/fce/doc1000.xml',
                     '../sandbox/fce/doc1002.xml',
                     '../sandbox/fce/doc1003.xml',
                     '../sandbox/fce/doc1005.xml',
                     '../sandbox/fce/doc1006.xml',
                     '../sandbox/fce/doc1008.xml']
        assert obtained == expected

    @attr("main")
    def test_mainrun(self):
        corpus_list, fileindexdict = read(corpus_dir=self.corpus_dir, output_dir=self.output_dir, working_dir=self.working_dir)
        # print pformat(corpus_list)
        # print pformat(fileindexdict)
        raise Exception

    @attr("rec")
    def test_recursive(self):
        cd = '../sandbox/fce2'
        corpus_list, fileindexdict = read(corpus_dir=cd, output_dir=cd, working_dir=cd)
        # print pformat(corpus_list)
        raise Exception

    def test_full(self):
        cd = '/Users/tuxedocat/cl/nldata/fce'
        od_wd = '../sandbox/fce3'
        corpus_list, fileindexdict = read(corpus_dir=cd, output_dir=od_wd, working_dir=od_wd)
        print pformat(corpus_list)
        raise Exception

class TestPreprocessor2:
    def setUp(self):
        import os, sys
        import cPickle as pickle
        import nltk
        self.corpus_dir = '../sandbox/fce'
        self.working_dir = '../sandbox/fce'
        self.output_dir = '../sandbox/fce'
        self.corpus_as_list, self.filelist = read(corpus_dir=self.corpus_dir, output_dir=self.output_dir, working_dir=self.working_dir)
    @attr("preprocess", "main")
    def test_preprocess1(self):
        preprocessor2.preprocess(self.corpus_as_list, self.filelist, modelist=["Incorrect_RV", "Gold"], destlist=[self.output_dir+"/gold.txt", self.output_dir+"/test.txt"])
        raise Exception
