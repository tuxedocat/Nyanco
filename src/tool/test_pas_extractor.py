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
from pprint import pformat
from pas_extractor import *
from nose.plugins.attrib import attr
from nose.tools import * 
#===============================================================================

class TestPasExtractor:
    def setUp(self):
        import os, sys
        import glob
        import collections
        relpath = '../../sandbox/pas/testdat*'
        self.testfile = glob.glob(relpath)
        self.eg1 = '../../sandbox/pas/afp_eng_201012_raw.parsed'
        self.eg2 = ['../../sandbox/pas/afp_eng_201012_raw.parsed', 
                    '../../sandbox/pas/afp_eng_201012_raw.parsed'] 
        self.testalt1 = "../../sandbox/pas/test_alt1.txt"

    @attr("extract_full")
    def test_extract_full1(self):
        pax = PEmod(self.testalt1)
        pax._extract_full(pax.raw)
        arg0list = [('13', 'we', '_', '_', 'PRP', '_', '15', 'nsubj', '_', '_', '_', '_', 'ARG0', '15')]
        arg1list = [('6', 'are', '_', '_', 'VBP', '_', '2', 'ccomp', '_', '_', 'be.01', '_', 'ARG1', '2'), 
                    ('17', 'right', '_', '_', 'NN', '_', '15', 'dobj', '_', '_', '_', '_', 'ARG1', '15'), 
                    ('22', 'life', '_', '_', 'NN', '_', '19', 'dobj', '_', '_', '_', '_', 'ARG1', '19'), 
                    ('28', 'into', '_', '_', 'IN', '_', '27', 'prep', '_', '_', 'into.2', '_', 'ARG1', '27')]
        assert pax.tmp_ARG0 == arg0list
        assert pax.tmp_ARG1 == arg1list


    @attr("extract_full")
    def test_extract_full2(self):
        pax = PEmod(self.testalt1)
        pax._extract_full(pax.raw)
        tagtuplelist = [('1', 'To', '_', '_', 'TO', '_', '2', 'infmark', '_', '_', '_', '_', '_', '_'),
                        ('2', 'sum', '_', '_', 'VB', '_', '0', 'ROOT', '_', '_', 'sum.01', '_', '_', '_'), 
                        ('3', 'up,', '_', '_', ',', '_', '6', 'punct', '_', '_', '_', '_', '_', '_'), 
                        ('4', 'famous', '_', '_', 'JJ', '_', '5', 'amod', '_', '_', '_', '_', '_', '_'), 
                        ('5', 'people', '_', '_', 'NNS', '_', '6', 'nsubj', '_', '_', '_', '_', '_', '_'), 
                        ('6', 'are', '_', '_', 'VBP', '_', '2', 'ccomp', '_', '_', 'be.01', '_', 'ARG1', '2'), 
                        ('7', 'the', '_', '_', 'DT', '_', '8', 'det', '_', '_', '_', '_', '_', '_'), 
                        ('8', 'same', '_', '_', 'JJ', '_', '6', 'cop', '_', '_', '_', '_', '_', '_'), 
                        ('9', 'as', '_', '_', 'IN', '_', '11', 'mark', '_', '_', '_', '_', '_', '_'), 
                        ('10', 'we', '_', '_', 'PRP', '_', '11', 'nsubj', '_', '_', '_', '_', '_', '_'), 
                        ('11', 'are,', '_', '_', 'VBP', '_', '6', 'advcl', '_', '_', '_', '_', '_', '_'), 
                        ('12', 'and', '_', '_', 'CC', '_', '11', 'cc', '_', '_', '_', '_', '_', '_'), 
                        ('13', 'we', '_', '_', 'PRP', '_', '15', 'nsubj', '_', '_', '_', '_', 'ARG0', '15'), 
                        ('14', 'all', '_', '_', 'DT', '_', '13', 'appos', '_', '_', '_', '_', '_', '_'), 
                        ('15', 'have', '_', '_', 'VBP', '_', '12', 'conj', '_', '_', 'have.03', '_', '_', '_'), 
                        ('16', 'the', '_', '_', 'DT', '_', '17', 'det', '_', '_', '_', '_', '_', '_'), 
                        ('17', 'right', '_', '_', 'NN', '_', '15', 'dobj', '_', '_', '_', '_', 'ARG1', '15'), 
                        ('18', 'to', '_', '_', 'TO', '_', '19', 'infmark', '_', '_', '_', '_', '_', '_'), 
                        ('19', 'lead', '_', '_', 'VB', '_', '17', 'infmod', '_', '_', 'lead.01', '_', '_', '_'), 
                        ('20', 'a', '_', '_', 'DT', '_', '22', 'det', '_', '_', '_', '_', '_', '_'), 
                        ('21', 'private', '_', '_', 'JJ', '_', '22', 'amod', '_', '_', '_', '_', '_', '_'), 
                        ('22', 'life', '_', '_', 'NN', '_', '19', 'dobj', '_', '_', '_', '_', 'ARG1', '19'), 
                        ('23', 'and', '_', '_', 'CC', '_', '22', 'cc', '_', '_', '_', '_', '_', '_'), 
                        ('24', 'no', '_', '_', 'DT', '_', '25', 'det', '_', '_', '_', '_', '_', '_'), 
                        ('25', 'right', '_', '_', 'NN', '_', '23', 'conj', '_', '_', '_', '_', '_', '_'), 
                        ('26', 'to', '_', '_', 'TO', '_', '27', 'infmark', '_', '_', '_', '_', '_', '_'), 
                        ('27', 'break', '_', '_', 'VB', '_', '25', 'infmod', '_', '_', 'break.02', '_', '_', '_'), 
                        ('28', 'into', '_', '_', 'IN', '_', '27', 'prep', '_', '_', 'into.2', '_', 'ARG1', '27'), 
                        ('29', 'their', '_', '_', 'PRP$', '_', '30', 'poss', '_', '_', 'SUBJECTIVE', '_', '_', '_'), 
                        ('30', 'privacy.', '_', '_', 'NN', '_', '28', 'pobj', '_', '_', '_', '_', '_', '_')]
        arg0list = [('13', 'we', '_', '_', 'PRP', '_', '15', 'nsubj', '_', '_', '_', '_', 'ARG0', '15')]
        arg1list = [('6', 'are', '_', '_', 'VBP', '_', '2', 'ccomp', '_', '_', 'be.01', '_', 'ARG1', '2'), 
                    ('17', 'right', '_', '_', 'NN', '_', '15', 'dobj', '_', '_', '_', '_', 'ARG1', '15'), 
                    ('22', 'life', '_', '_', 'NN', '_', '19', 'dobj', '_', '_', '_', '_', 'ARG1', '19'), 
                    ('28', 'into', '_', '_', 'IN', '_', '27', 'prep', '_', '_', 'into.2', '_', 'ARG1', '27')]
        relationslist = [('15', 'have', '_', '_', 'VBP', '_', '12', 'conj', '_', '_', 'have.03', '_', '_', '_'),
                         ('2', 'sum', '_', '_', 'VB', '_', '0', 'ROOT', '_', '_', 'sum.01', '_', '_', '_'),
                         ('15', 'have', '_', '_', 'VBP', '_', '12', 'conj', '_', '_', 'have.03', '_', '_', '_'),
                         ('19', 'lead', '_', '_', 'VB', '_', '17', 'infmod', '_', '_', 'lead.01', '_', '_', '_'),
                        ('27', 'break', '_', '_', 'VB', '_', '25', 'infmod', '_', '_', 'break.02', '_', '_', '_')]
        pasdic_list = [{"PRED":("have", "VBP","conj","have.03"), "ARG0":("we", "PRP", "nsubj", "_"), "ARG1":("right", "NN", "dobj", "_")},
                        {"PRED":("sum", "VB", "ROOT", "sum.01"), "ARG0":None, "ARG1":("are", "VBP", "ccomp", "be.01")},
                        {"PRED":("lead", "VB", "infmod", "lead.01"), "ARG0":None, "ARG1":("life", "NN", "dobj", "_") },
                        {"PRED":("break", "VB", "infmod", "break.02"), "ARG0":None, "ARG1":("into", "IN", "prep", "into.2")} ]
        print pax.pasdic_list
        assert_items_equal(pasdic_list,pax.pasdic_list)


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

    @attr("egtest")
    def test_extract_gigaword_single(self):
        '''test for real Gigaword data but single'''
        pax = PasExtractor(self.eg1)
        result = pax.extract()
        triples = collections.Counter(result)
        # print triples.most_common(1)
        self.eg_single = triples
        logging.debug(pformat(triples))
        raise AssertionError

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
    @attr("egtest")
    def test_tsvout_huge(self):
        '''check whether the tsv-output is correct'''
        pax = PasExtractor(self.eg1)
        result = pax.extract()
        triples = collections.Counter(result)
        input_dir = '../../sandbox/pas'
        prefix = 'test_tsvout_huge_'
        output2file(input_dir, prefix, triples)
        output = open(os.path.join(input_dir, 'test_tsvout_huge_PAS.tsv'),'r').read()