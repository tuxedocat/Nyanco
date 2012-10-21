#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/feature_extractor.py
Created on 18 Oct. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from numpy import array
from pattern.text import en
from nltk.corpus import wordnet as wn

try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *


class FeatureExtractorBase(object):
    def __init__(self, tags=[], verb=""):
        self.features = defaultdict(float)
        self.v = verb
        self.gen_fn = lambda x,y,z: "_".join([x, str(y), z])
        self.nullfeature = {"NULL":1}
        try:
            self.tags = [t for t in tags if not t is ""]
            if len(tags[0].split("\t")) == 14:
                self.col_suf = 1
                self.col_pos = 4
            elif len(tags[0].split("\t")) == 8:
                self.col_suf = 1
                self.col_pos = 2
            else:
                print "FeatureExtractor: initializing... \nUnknown input format"
                raise IndexError
            try:
                self.SUF = [line.split("\t")[self.col_suf] for line in self.tags]
                self.SUF_l = [en.conjugate(line.split("\t")[self.col_suf], tense="infinitive") for line in self.tags]
                self.POS = [line.split("\t")[self.col_pos] for line in self.tags]
                self.WL = zip(self.SUF, self.POS)
                self.v_idx = self._find_verb_idx()
            except Exception, e:
                print pformat(e)
                print tags
        except AttributeError:
            if len(tags[0]) == 14:
                self.col_suf = 1
                self.col_pos = 4
            elif len(tags[0]) == 8:
                self.col_suf = 1
                self.col_pos = 2
            self.tags = tags
            try:
                self.SUF = [t[self.col_suf] for t in self.tags]
                self.SUF_l = [en.conjugate(t[self.col_suf], tense="infinitive") for t in self.tags]
                self.POS = [t[self.col_pos] for t in self.tags]
                self.WL = zip(self.SUF, self.POS)
                self.v_idx = self._find_verb_idx()
            except Exception, e:
                print pformat(e)
                print tags
        except Exception, e:
            print pformat(e)
            print tags
            self.features.update(self.nullfeature)


    def _find_verb_idx(self):
        verbpos = [idx for idx, sufpos in enumerate(zip(self.SUF,self.POS)) if sufpos[0] == self.v and "VB" in sufpos[1]]
        if verbpos:
            return verbpos[0]
        else:
            verbpos = [idx for idx, sufpos in enumerate(zip(self.SUF_l,self.POS)) if sufpos[0] == self.v and "VB" in sufpos[1]]
            if verbpos:
                return verbpos[0]
            else:
                verbpos = int(len(self.SUF)/2)
                return verbpos

    @classmethod
    def read_corpusfiles(self, corpuspath=""):
        """
        This classmethod reads corpus (pickled files, separated by each verbs)
        from the given directory, 
        returns a dictionary for next process
        """
        corpusdict = defaultdict(list)


    def save_to_file(self):
        raise NotImplementedError


class SimpleFeatureExtractor(FeatureExtractorBase):
    """
    Extract features for given case, 
    this one has Surface/POS ngram, dependency, Semantic tags, and SRL related features 
    introduced by Liu et.al 2010

    More interesting features such as sentence wise topic models will be implemented in next update
    """

    # @classmethod
    def ngrams(self, n=5, v_idx=None):
        """
        Make a query for ngram frequency counter
        @takes:
            n :: N gram size (if n=5, [-2 -1 word +1 +2])
            v_idx:: int, positional index of the checkpoint
            w_list:: list, words of a sentence
            alt_candidates:: list, alternative candidates if given
        @returns:
            suf_ngram: {"suf_-2_the": 1, "suf_-1_cat": 1, "suf_0_eats": 1,...}
            pos_ngram: {"suf_-2_DT": 1, "suf_-1_NN": 1, "suf_0_VBZ": 1,...}
        """
        try:
            if not v_idx:
                v_idx = self.v_idx
            suf_ngram = {}
            pos_ngram = {}
            window = int((n - 1)/2)
            if not v_idx:
                v_idx = 0
            core = self.WL[v_idx]
            _left = [word for index, word in enumerate(self.WL) if index < v_idx][-window:]
            _right = [word for index, word in enumerate(self.WL) if index > v_idx][:window]
            concat = _left + [core] + _right
            suf_ngram = {self.gen_fn("SUF", i-window, w[0]):1 for i, w in enumerate(concat)}
            pos_ngram = {self.gen_fn("POS", i-window, w[1]):1 for i, w in enumerate(concat)}
            self.features.update(suf_ngram)
            self.features.update(pos_ngram)
        except Exception, e:
            print pformat(e)
            self.features.update(self.nullfeature)

    @classmethod
    def read_wordnet_tag(self):
        raise NotImplementedError

    @classmethod
    def convert_to_scipy(self, casedict={}):
        raise NotImplementedError


