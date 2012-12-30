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
import sys
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from numpy import array
from pattern.text import en
import traceback
from nltk import ngrams as ng
try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *


class BCluster(object):
    bcdic = pickle.load(open("../sandbox/bc_256.pkl2", "rb"))

    def getbits(self, w):
        try:
            _bits = BCluster.bcdic[w]
        except KeyError:
            try:
                _bits = BCluster.bcdic[w.lower()]
            except KeyError:
                _bits = None
        except:
            raise
        return _bits


class FeatureExtractorBase(object):
    nullfeature = {"NULL":1}
    VE_count = 0

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
        cls.col_srlrel = 12 
        cls.col_srl = 13

    @classmethod
    def set_col_r(cls):
        cls.conll_type = "reduced"
        cls.col_suf = 1
        cls.col_pos = 2
        cls.col_headid = 4
        cls.col_deprel = 3
        cls.col_netag = 5
        cls.col_srlrel = 6
        cls.col_srl = 7

    def __init__(self, tags=[], verb="", v_idx=None):
        self.features = defaultdict(float)
        self.v = verb
        try:
            # for extracting features from parsed data (tab separated dataset in CoNLL like format)
            self.tags = [t.split("\t") for t in tags if not t is ""]
            _t = len(self.tags[0])
            # print "FeatureExtractor: Num of column of tags is %d"%_t
            if _t == 14:
                FeatureExtractorBase.set_col_f()
            elif _t == 8:
                FeatureExtractorBase.set_col_r()
        except AttributeError, IndexError:
            # for extracting features from tags' list
            self.tags = tags
            _t = len(self.tags[0])
            # print "FeatureExtractor: Num of column of tags is %d"%_t
            if _t == 14:
                FeatureExtractorBase.set_col_f()
            elif _t == 8:
                FeatureExtractorBase.set_col_r()
        except:
            print verb
            print pformat(tags)
            raise
        try:
            self.SUF = [t[FeatureExtractorBase.col_suf].lower() for t in self.tags]
            self.POS = [t[FeatureExtractorBase.col_pos] for t in self.tags]
            self.WL = zip(self.SUF, self.POS)
            self.v_idx = self._find_verb_idx() if not v_idx else v_idx
            if self.v_idx is None:
                FeatureExtractorBase.VE_count += 1
                raise ValueError
            else:
                pass
                # print "verb is ", tags[self.v_idx]
        except Exception, e:
            # print pformat(["FeatureExtractor: ", e])
            # traceback.print_exc(file=sys.stdout)
            # print tags[0]
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

    """

    def ngrams(self, n=5, v_idx=None):
        """
        Make a query for ngram frequency counter
        @takes:
            n :: N gram size (if n=5, [-2 -1 +1 +2])
            v_idx:: int, positional index of the checkpoint
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
            _lefts = [word for index, word in enumerate(self.SUF) if index < v_idx and index != v_idx][-window:]
            _leftp = [word for index, word in enumerate(self.POS) if index < v_idx and index != v_idx][-window:]
            _rights = [word for index, word in enumerate(self.SUF) if index > v_idx and index != v_idx][:window]
            _rightp = [word for index, word in enumerate(self.POS) if index > v_idx and index != v_idx][:window]
            concats = _lefts + ["*V*"] + _rights
            concatp = _leftp + ["*V*"] + _rightp
            suf_unigram = {SimpleFeatureExtractor.gen_fn(["SUF1G", str(i-window), "".join(w)]):1 
                        for i, w in enumerate(concats) if w != "*V*"}
            pos_unigram = {SimpleFeatureExtractor.gen_fn(["POS1G", str(i-window), "".join(w)]):1 
                        for i, w in enumerate(concatp) if w != "*V*"}
            suf_bigram = {SimpleFeatureExtractor.gen_fn(["SUF2G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concats, 3)) if w[0] == "*V*" or w[2] == "*V*"} if n >= 5 else {}
            pos_bigram = {SimpleFeatureExtractor.gen_fn(["POS2G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concatp, 3)) if w[0] == "*V*" or w[2] == "*V*"} if n >= 5 else {}
            suf_trigram = {SimpleFeatureExtractor.gen_fn(["SUF3G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concats, 4)) if w[0] == "*V*" or w[3] == "*V*"} if n >= 7 else {}
            pos_trigram = {SimpleFeatureExtractor.gen_fn(["POS3G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concatp, 4)) if w[0] == "*V*" or w[3] == "*V*"} if n >= 7 else {}
            suf_c3gram = {SimpleFeatureExtractor.gen_fn(["SUF3G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concats, 3)) if w[1] == "*V*"} if n >= 3 else {}
            # suf_c5gram = {SimpleFeatureExtractor.gen_fn(["SUF5G", "", "-".join(w)]):1 
            #             for i, w in enumerate(ng(concats, 5)) if w[2] == "*V*"} if n >= 5 else {}
            # suf_c7gram = {SimpleFeatureExtractor.gen_fn(["SUF7G", "", "-".join(w)]):1 
                        # for i, w in enumerate(ng(concats, 7)) if w[3] == "*V*"} if n >= 7 else {}
            pos_c3gram = {SimpleFeatureExtractor.gen_fn(["POS3G", "", "-".join(w)]):1 
                        for i, w in enumerate(ng(concatp, 3)) if w[1] == "*V*"} if n >= 3 else {}
            # pos_c5gram = {SimpleFeatureExtractor.gen_fn(["POS5G", "", "-".join(w)]):1 
            #             for i, w in enumerate(ng(concatp, 5)) if w[2] == "*V*"} if n >= 5 else {}
            # pos_c7gram = {SimpleFeatureExtractor.gen_fn(["POS7G", "", "-".join(w)]):1 
                        # for i, w in enumerate(ng(concatp, 7)) if w[3] == "*V*"} if n >= 7 else {}
            self.features.update(suf_unigram)
            self.features.update(pos_unigram)
            self.features.update(suf_bigram)
            self.features.update(pos_bigram)
            self.features.update(suf_trigram)
            self.features.update(pos_trigram)
            self.features.update(suf_c3gram)
            # self.features.update(suf_c5gram)
            # self.features.update(suf_c7gram)
            self.features.update(pos_c3gram)
            # self.features.update(pos_c5gram)
            # self.features.update(pos_c7gram)
        except Exception, e:
            pass
            # self.features.update(SimpleFeatureExtractor.nullfeature)

    def chunk(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            l_ctxid = [idx for idx, pt in enumerate(self.POS) if idx < v_idx and pt.startswith("NN")]
            r_ctxid = [idx for idx, pt in enumerate(self.POS) if idx > v_idx and pt.startswith("NN")]
            l_nearestNN = {SimpleFeatureExtractor.gen_fn(["NN", "L", self.SUF[l_ctxid[-1]]]) : 1} if l_ctxid else None
            r_nearestNN = {SimpleFeatureExtractor.gen_fn(["NN", "R", self.SUF[r_ctxid[0]]]) : 1} if r_ctxid else None
            self.features.update(l_nearestNN)
            self.features.update(r_nearestNN)
        except:
            pass



class FeatureExtractor(SimpleFeatureExtractor):
    def dependency(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            deps = [(t[FeatureExtractor.col_deprel], t[FeatureExtractor.col_suf], 
                     t[FeatureExtractor.col_pos], t[FeatureExtractor.col_netag]) for t in self.tags
                     if int(t[FeatureExtractor.col_headid]) == v_idx+1]
            # depr = {FeatureExtractor.gen_fn(["DEP", d[0].upper(), d[1].lower()+"/"+d[2]]):1 for d in deps}
            depp = {FeatureExtractor.gen_fn(["DEP", d[0].upper(), d[1].lower()]):1 for d in deps}
            depn = {FeatureExtractor.gen_fn(["DEP", d[0].upper(), d[3]]):1 for d in deps if not d[3]=="_" }
            # self.features.update(depr)
            self.features.update(depp)
            self.features.update(depn)
        except Exception, e:
            logging.debug(pformat(e))
            # self.features.update(FeatureExtractor.nullfeature)


    def ne(self, v_idx=None):
        try:
            if not v_idx:
                v_idx = self.v_idx
            ne = self.tags[v_idx][FeatureExtractor.col_netag]
            ne_tag = {"V-NE_" + ne: 1} if not ne == "_" else {}
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
            ARGS = [(l[FeatureExtractor.col_srlrel], l[FeatureExtractor.col_suf], 
                     l[FeatureExtractor.col_pos], l[FeatureExtractor.col_netag]) for l in self.tags 
                    if l[FeatureExtractor.col_srl] != "_" and int(l[FeatureExtractor.col_srl]) - 1 == v_idx]
            if ARGS:
                srlf = {FeatureExtractor.gen_fn(["SRL", t[0], en.lemma(t[1])]):1 for t in ARGS}
                # srlp = {FeatureExtractor.gen_fn(["SRL", t[0], en.lemma(t[1])+"/"+t[2]]):1 for t in ARGS}
                srln = {FeatureExtractor.gen_fn(["SRL", t[0], t[3]]):1 for t in ARGS if not t[3]=="_"}
                self.features.update(srlf)
                # self.features.update(srlp)
                self.features.update(srln)
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
