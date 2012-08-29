#! /usr/bin/env python
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
#=====================================================================================================

class TestCorpusReader:
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


    def mainrun_test(self):
        corpus_list, fileindexlist = read(corpus_dir=self.corpus_dir, output_dir=self.output_dir, working_dir=self.working_dir)
        print pformat(corpus_list[0:3])
        print pformat(fileindexlist)
        raise Exception