#!/usr/bin/env python
# coding: utf-8
'''
Nyanco/src/tool/altword_generator.py
Created on 12 Oct 2012


Retrieve alternative words for given word using wordnet and verbnet
via nltk

'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

import cPickle as pickle
from pprint import pformat
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from nose.plugins.attrib import attr
from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
from pattern.text import en
import collections

class AlternativeGenerator(object):
    """
    Takes surface of a word and its wordnet synset name if given, 
    Returns alternative candidates as a list (10 alternatives is maximum as default setting)
    """
    def __init__(self, suf="", wncat="", maxnum=20, pos="VB", include_hyponyms=False, include_uncertain=False, score=True):
        self.surface = suf
        self.pos = pos
        self.wncat = wncat
        self.maxnum_cand = maxnum
        self.include_hyponyms = include_hyponyms
        self.include_uncertain = include_uncertain
        self.alternatives = []
        self.score = score


    def generate_from_wordnet(self):
        self.synsetnames = []
        if self.wncat != "" and len(self.wncat.split(".")) == 2:
            self.synsetnames.append(self._mk_synsetname())
        elif self.include_uncertain:
            self.synsetnames = wn.synsets(self.surface, pos=self._posmap())
        for synset in self.synsetnames:
            if self.score:
                tmp = [(w[0], w[2]) for w in self._traverse_synsets(synset=synset)]
            else:
                tmp = [(w[0], 0) for w in self._traverse_synsets(synset=synset)]
            for alt in tmp:
                if len(self.alternatives) >= self.maxnum_cand:
                    break
                elif alt in self.alternatives:
                    pass
                else:
                    self.alternatives.append(alt)
        return list(set(self.alternatives))


    def _traverse_synsets(self, synset=None, depth=1):
        """
        starting from given synset, traverse synsets.
        firstly, take hypernym of given synset, then try to find hyponyms 
        """
        try:
            hyp = synset.hypernyms()[0]
            hyponyms = hyp.hyponyms()
            syns = [(unicode(s.name.split(".")[0]), s, synset.lch_similarity(s) ) for s in hyponyms]
            syns = sorted(syns, key=lambda x: x[2], reverse=True)
        except IndexError, e:
            # logging.debug(pformat(e))
            hyp = []
            syns = [] 
        return syns


    def _posmap(self):
        return "v" if "VB" in self.pos else "n"

    def _mk_synsetname(self, word=""):
        p = self._posmap()
        tmp = word.split(u".")
        tmp.insert(1, p)
        return ".".join(tmp)


    def _include_uncertain(self):
        pass


class AlternativeReader(object):
    def __init__(self, verbset_path=""):
        try:
            verbset = pickle.load(open(verbset_path, "rb"))
            self.altdic = verbset
        except Exception, e:
            print e
            raise

    def get_altwordlist(self, verb=""):
        try:
            altlist = [wt[0] for wt in self.altdic[verb]]
        except KeyError:
            altlist = AlternativeGenerator(suf=verb, wncat="", include_uncertain=True).generate_from_wordnet()
        return altlist

    @classmethod
    def get_lemma(self, verb=""):
        return en.conjugate(verb, tense="infinitive")

class TestAltGen(object):
    """
    Test class for nosetests of AlternativeGenerator
    """
    def setUp(self):
        self.word1 = "explain"
        self.word1_s = "explain.v.01"
        self.uncertainwords = []
        self.alt_dict_path = "./ranked_alt20.pickle2"

    def test_given_synset(self):
        alt = AlternativeGenerator(suf=self.word1, wncat="", include_uncertain=True).generate_from_wordnet()
        print alt
        raise Exception

    def test_alt_reader(self):
        rdr = AlternativeReader(self.alt_dict_path)
        altlist = rdr.get_altwordlist("explain")
        print altlist
        raise Exception


    def test_uncertain_synset(self):
        pass


def gen_WNCS(CS):
    CS_WN = collections.OrderedDict()
    voc = CS.keys()
    for v in voc:
        _ag = AlternativeGenerator(v, maxnum=100, score=False, include_hyponyms=True, include_uncertain=True, pos="VB")
        ls = _ag.generate_from_wordnet()
        ls = [_v for _v in ls if _v[0] in voc]
        if ls:
            CS_WN[v] = ls
        else:
            pass
    return CS_WN