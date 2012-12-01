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

try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *


class FeatureExtractorBase(object):
    nullfeature = {"NULL":0}
    conll_type = "full"
    col_suf = 1
    col_pos = 4
    col_headid = 6
    col_deprel = 7
    col_netag = 10
    col_srlrel = 12
    col_srl = 13

    @classmethod
    def gen_fn(cls, l=None):
        return "_".join(l)

    @classmethod
    def set_col_f(cls):
        cls.conll_type = "full"
        cls.col_suf = 1
        cls.col_pos = 4
        cls.col_headid = 6
        cls.col_deprel = 7
        cls.col_netag = 10
        cls.col_srl = 12
        cls.col_srlrel = 13

    @classmethod
    def set_col_r(cls):
        cls.conll_type = "reduced"
        cls.col_suf = 1
        cls.col_pos = 2
        cls.col_headid = 4
        cls.col_deprel = 3
        cls.col_netag = 5
        cls.col_srl = 6
        cls.col_srlrel = 7

    def __init__(self, tags=[], verb="", v_idx=None, conll_type="full"):
        try:
            assert conll_type == self.__class__.conll_type
        except AssertionError:
            if conll_type == "reduced":
                FeatureExtractorBase.set_col_r()
            elif conll_type == "full":
                FeatureExtractorBase.set_col_f()
        self.features = defaultdict(float)
        self.v = verb
        try:
            # for extracting features from parsed data (tab separated dataset in CoNLL like format)
            self.tags = [t.split("\t") for t in tags if not t is ""]
        except AttributeError:
            # for extracting features from tags' list
            self.tags = tags
        try:
            self.SUF = [t[FeatureExtractorBase.col_suf] for t in self.tags]
            self.POS = [t[FeatureExtractorBase.col_pos] for t in self.tags]
            self.WL = zip(self.SUF, self.POS)
            self.v_idx = self._find_verb_idx() if not v_idx else v_idx
            if self.v_idx is None:
                raise ValueError
            else:
                pass
                # print "verb is ", tags[self.v_idx]
        except Exception, e:
            print pformat(e)
            # print pformat(tags)
            # logging.debug(pformat(tags))
            self.features.update(FeatureExtractorBase.nullfeature)


    def _find_verb_idx(self):
        verbpos = [idx for idx, sufpos in enumerate(zip(self.SUF, self.POS)) if sufpos[0] == self.v and "VB" in sufpos[1]]
        if verbpos:
            return verbpos[0]
        else:
            SUF_l = [en.lemma(w) for w in self.SUF]
            verbpos = [idx for idx, sufpos in enumerate(zip(SUF_l, self.POS)) if sufpos[0] == self.v and "VB" in sufpos[1]]
            if verbpos:
                return verbpos[0]
            else:
                # verbpos = int(len(self.SUF)/2)
                return None

    @classmethod
    def read_corpusfiles(self, corpuspath=""):
        """
        This classmethod reads corpus (pickled files, separated by each verbs)
        from the given directory, 
        returns a dictionary for next process
        """
        corpusdict = defaultdict(list)
        raise NotImplementedError


    def save_to_file(self):
        raise NotImplementedError


class SimpleFeatureExtractor(FeatureExtractorBase):
    """
    Extract features for given case, 
    this one has Surface/POS ngram, dependency, Semantic tags, and SRL related features 
    introduced by Liu et.al 2010

    More interesting features such as sentence wise topic models will be implemented in next update
    """

    def ngrams(self, n=5, v_idx=None):
        """
        Make a query for ngram frequency counter
        @takes:
            n :: N gram size (if n=5, [-2 -1 +1 +2])
            v_idx:: int, positional index of the checkpoint
            w_list:: list, words of a sentence
            alt_candidates:: list, alternative candidates if given
        @returns:
            suf_ngram: {"suf_-2_the": 1, "suf_-1_cat": 1, ...}
            pos_ngram: {"suf_-2_DT": 1, "suf_-1_NN": 1, ...}
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
            _left = [word for index, word in enumerate(self.WL) if index < v_idx and index != v_idx][-window:]
            _right = [word for index, word in enumerate(self.WL) if index > v_idx and index != v_idx][:window]
            concat = _left + ["__CORE__"] + _right
            suf_ngram = {SimpleFeatureExtractor.gen_fn(["SUF", str(i-window), w[0]]):1 for i, w in enumerate(concat) if w != "__CORE__"}
            pos_ngram = {SimpleFeatureExtractor.gen_fn(["POS", str(i-window), w[1]]):1 for i, w in enumerate(concat) if w != "__CORE__"}
            self.features.update(suf_ngram)
            self.features.update(pos_ngram)
        except Exception, e:
            print pformat(e)
            # self.features.update(SimpleFeatureExtractor.nullfeature)

    def chunk(self):
        raise NotImplementedError



class FeatureExtractor(SimpleFeatureExtractor):
    def dependency(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            deps = [(t[FeatureExtractor.col_deprel], t[FeatureExtractor.col_suf]) for t in self.tags
                    if int(t[FeatureExtractor.col_headid]) == v_idx+1]
            depr = {FeatureExtractor.gen_fn(["DEP", d[0].upper(), d[1]]):1 for d in deps}
            self.features.update(depr)
        except Exception, e:
            logging.debug(pformat(e))
            # self.features.update(FeatureExtractor.nullfeature)


    def ne(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            ne = self.tags[v_idx][FeatureExtractor.col_netag]
            if not ne == "_":
                ne_tag = {"V-NE_" + ne: 1}
            else:
                ne_tag = {}
            self.features.update(ne_tag)
        except Exception, e:
            logging.debug(pformat(e))

    @classmethod
    def __format_srl(cls, srldic):
        srl= []
        moc = ("","","","")
        for pkey in srldic:
            out = {}
            out["PRED"] = (pkey[cls.col_suf], pkey[cls.col_pos], pkey[cls.col_deprel], pkey[cls.col_netag])
            try:
                a0 = srldic[pkey]["ARG0"]
                out["ARG0"] = (a0[cls.col_suf], a0[cls.col_pos], a0[cls.col_deprel], a0[cls.col_netag])
            except KeyError:
                out["ARG0"] = moc 
            try:
                a1 = srldic[pkey]["ARG1"]
                out["ARG1"] = (a1[cls.col_suf], a1[cls.col_pos], a1[cls.col_deprel], a1[cls.col_netag])
            except KeyError:
                out["ARG1"] = moc 
            srl.append(out)
        return srl


    def srl(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            self.tmp_ARG0 = []
            self.tmp_ARG1 = []
            self.tmp_PRED = defaultdict(dict)
            ARGS = [(l[FeatureExtractor.col_srlrel], l[FeatureExtractor.col_suf]) for l in self.tags 
                    if l[FeatureExtractor.col_srl] != "_" and int(l[FeatureExtractor.col_srl]) - 1 == v_idx]
            if ARGS:
                srlf = {FeatureExtractor.gen_fn(["SRL", t[0], en.lemma(t[1])]):1 for t in ARGS}
                self.features.update(srlf)
        except Exception, e:
            logging.debug(pformat(e))
            # self.features.update(FeatureExtractor.nullfeature)

    @classmethod
    def _load_errorprobs(cls, vspath=None):
        if vspath:
            cls.dic_errorprobs = pickle.load(open(vspath, "rb"))
        else:
            raise IOError


    def _read_errorprob(self):
        try:
            prob_v = FeatureExtractor.dic_errorprobs[self.v]
        except KeyError:
            prob_v = FeatureExtractor.dic_errorprobs[en.lemma(self.v)]
        finally:
            pass

    def errorprob(self, vspath=None):
        if FeatureExtractor.dic_errorprobs:
            pass
        else:
            try:
                _load_errorprobs(vspath)
            except IOError:
                pass


    def topic(self):
        raise NotImplementedError