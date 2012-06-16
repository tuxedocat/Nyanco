#!/usr/bin/env python
# coding:utf-8
__author__ = 'yu-s'
__version__ = '0.001'
__licence__ = 'free4all'
__description__ = '''
Count freq. on LM
'''
# from gensim import corpora, models, similarities, utils
# import cPickle as pickle
from collections import defaultdict
import random 
from irstlm import getSentenceScore 


class Webfrequency(object):
    '''
    this class is for estimating frequency of given list of queries 
    on the Web1T corpus using IRSTLM
    '''
    def __init__(self, irstlm_LM, dict_queries, dict_id2words):
        '''
        Constructor args:
            irstlm_LM: object created by initLM
            dict_queries: 
            dict_id2words: {0:'give',1:'offer',....}

        '''
        self.lm = irstlm_LM 
        self.dict_queries = dict_queries
        self.resultdict = defaultdict(list)
        self.id2words = dict_id2words


    def countall(self):
        '''Return score on LM for queries
        each query is given as a list: [w-2, w-1, cand, w1, w2]
        ARGS:
            Nothing
        RETURNS:
            scores on LM for each queries
        '''
        for wid in self.dict_queries:
            qlist = self.dict_queries[wid][:]
            for ql in qlist:
                correct = self.id2words[wid]
                eachresult = {}
                eachresult.update({'correct':correct, 'rawtext': " ".join(ql)})
                for candid in self.id2words:
                    cand = self.id2words[candid] 
                    q_cand = ql[:]
                    try:
                        # c_index = q_cand.index(cand)
                        c_index = 2
                        q_cand.pop(c_index)
                        q_cand.insert(c_index, cand)
                    except:
                        print q_cand
                        pass
                    query4LM  = " ".join(q_cand)
                    count = getSentenceScore(self.lm, query4LM)
                    eachresult.update({cand:count})
                self.resultdict[wid].append(eachresult)
        print self.resultdict.items()
