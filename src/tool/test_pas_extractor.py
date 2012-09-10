#! /usr/bin/env python
# coding: utf-8
'''
nyanco/tool/test_pas_extractor.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import os
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
import collections
from pas_extractor import *
#===============================================================================

class TestPasExtractor:
    def setUp(self):
        import os, sys
        import glob
        import collections
        relpath = '../sandbox/pas/testdat*'
        self.testfile = glob.glob(relpath)
        self.eg1 = '../sandbox/pas/afp_eng_201012_raw.parsed'
        self.eg2 = ['../sandbox/pas/afp_eng_201012_raw.parsed', 
                    '../sandbox/pas/afp_eng_201012_raw.parsed'] 

    def test_extract1(self):
        pax = PasExtractor(self.testfile[0])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('loved', 'We', 'cat')])
        assert triples == expected

    def test_extract2(self):
        pax = PasExtractor(self.testfile[1])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'dog')])
        assert triples == expected

    def test_extract3(self):
        '''test for complicated structure (which will be ignored)'''
        pax = PasExtractor(self.testfile[2])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('urged', 'Royce', 'House'), ('insisted', 'Burns', 'doing')])
        assert triples == expected

    def test_extract_multi(self):
        '''test for multiple files'''
        triples = collections.Counter()
        for f in self.testfile[3:7]:
            pax = PasExtractor(f)
            result = pax.extract()
            tmpc = collections.Counter(result)
            triples = triples + tmpc
        expected = collections.Counter([('urged', 'Royce', 'House'), ('insisted', 'Burns', 'doing'),
                                        ('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat')])
        assert triples == expected

    def test_extract_gigaword_single(self):
        '''test for real Gigaword data but single'''
        pax = PasExtractor(self.eg1)
        result = pax.extract()
        triples = collections.Counter(result)
        # print triples.most_common(1)
        self.eg_single = triples

    def test_extract_gigaword_multi(self):
        '''test for multiple files of Gigaword data'''
        triples = collections.Counter()
        for f in self.eg2:
            pax = PasExtractor(f)
            result = pax.extract()
            tmpc = collections.Counter(result)
            triples = triples + tmpc
        self.eg_multi = triples
        # print self.eg_multi

    def test_tsvout(self):
        '''check whether the tsv-output is correct'''
        pax = PasExtractor(self.testfile[1])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'dog')])
        input_dir = '../sandbox/pas'
        prefix = 'test_tsvout_'
        output2file(input_dir, prefix, triples)
        expected = open(os.path.join(input_dir, 'correct1.tsv'),'r').read()
        output = open(os.path.join(input_dir, 'test_tsvout_PAS.tsv'),'r').read()
        assert output == expected

    def test_tsvout_huge(self):
        '''check whether the tsv-output is correct'''
        pax = PasExtractor(self.eg1)
        result = pax.extract()
        triples = collections.Counter(result)
        input_dir = '../sandbox/pas'
        prefix = 'test_tsvout_huge_'
        output2file(input_dir, prefix, triples)
        output = open(os.path.join(input_dir, 'test_tsvout_huge_PAS.tsv'),'r').read()