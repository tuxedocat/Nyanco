#coding: utf-8
'''
Nyanco/src/suggest_each.py
Created on 25 Jan. 2013
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
import os
import glob
import traceback
import progressbar
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
import numpy as np
from feature_extractor import SentenceFeatures, proc_easyadapt
from tool.sparse_matrices import *
from classifier import *
from tool.sennaparser import *


def suggest_for_testset(corpuspath="", cspath="", 
                        modelrootpath="", modeltype="", features=[]):
    results = []
    with open(corpuspath) as f:
        testdata = pickle.load(f)
    with open(cspath) as f:
        confusionsets = pickle.load(f)
    sennapath = unicode(os.environ["SENNAPATH"])
    parser = SennaParser(sennapath)
    for case in testdata:
        try:
            result = suggestone(confusionsets, case, modelrootpath, modeltype, features, parser)
            results.append(result)
        except AssertionError:
            print "AssertionError in cp:: \n", pformat(case)
            results.append( (0, 0.0, None) )
        except Exception, e:
            print pformat(e)
            results.append( (0, 0.0, None) )
    return results


def suggestone(cs=None, case=None, modelroot="", modeltype="", features=None, parser=None):
    """
    Parameters
    ----------
    cs: dict like object (values are lists)
        confusion set(s)
    case: tuple of (testsent-as-list, v_idx, incorrect_inf, correct_inf, correctsentence)
        testcase
    modelroot: string
        path to root dir. of models
    features: list
        e.g., ["5gram", "chunk"]
    """
    raw_sent = case[0]
    v_idx = case[1]
    setname = case[2]
    gold = case[3]
    assert setname in cs
    parsed = parser.parseSentence(" ".join(raw_sent))
    features = get_features(parsed, setname, v_idx, features)
    # print pformat(features)
    model, fmap, label2id = _load_model(setname, modelroot, modeltype)
    if model and fmap and label2id:
        ic_clsid = label2id[setname]
        X = fmap.transform(features)
        Y = label2id[gold] if gold in label2id else None
        probdist = model.predict_prob(X)[0]
        if probdist is not None:
            detected, RR, suggestion = _kbest_rank_detector(probdist, 5, ic_clsid, Y)
            suggestion = _postprocess_suggestion(suggestion, label2id)
            return detected, RR, suggestion



def _kbest_rank_detector(probdist=None, k=5, orgidx=None, goldidx=None):
    try:
        probdist = probdist.tolist()
        _k = int(float(len(probdist)*(float(k)/50))) + 1
        probs = [(i, p) for i, p in enumerate(probdist)]
        probs.sort(key=lambda x: x[1], reverse=True)
        suggestion = probs[:k]
        rank_org = [i for i, t in enumerate(probs) if t[0] == orgidx][0] + 1
        try:
            rank_gold = [i for i, t in enumerate(probs) if t[0] == goldidx][0] + 1
            RR = float(1.0/rank_gold)
        except:
            RR = 0.0
        return (1, RR, suggestion) if rank_org > k else (0, RR, suggestion)
    except IndexError:
        RR = 0.0
        return (0, RR, suggestion)
        # raise WordNotInCsetError


def _load_model(setname="", modelroot="", modeltype=""):
    m = None
    fm = None
    l2id = None
    try:
        modelpath = os.path.join(modelroot, setname)
        # print os.path.join(modelroot,"model_"+self.modeltype+".pkl2")
        with open(os.path.join(modelpath,"model_"+modeltype+".pkl2"), "rb") as mf:
            m = pickle.load(mf)
        with open(os.path.join(modelpath,"featuremap.pkl2"), "rb") as mf:
            fm = pickle.load(mf)
        with open(os.path.join(modelpath,"label2id.pkl2"), "rb") as mf:
            l2id = pickle.load(mf)
    except Exception, e:
        pass
    return m, fm, l2id


def _postprocess_suggestion(suggestions=None, label2id=None):
    try:
        id2l = {v:k for k,v in label2id.iteritems()}
        named = [(id2l[i], p) for i, p in suggestions]
    except:
        named = None
    finally:
        return named


def get_features(tags=[], v="", v_idx=None, features=[]):
    fe = SentenceFeatures(tags=tags, verb=v, v_idx=v_idx)
    if "chunk" in features:
        fe.chunk()
    if "3gram" in features:
        fe.ngrams(n=3)
    if "5gram" in features:
        fe.ngrams(n=5)
    if "7gram" in features:
        fe.ngrams(n=7)
    if "dependency" in features:
        fe.dependency()
    if "ne" in features:
        fe.ne()
    if "srl" in features:
        fe.srl()
    if "topic" in features:
        fe.topic()
    if "errorprob" in features:
        fe.ep()
    # print pformat(fe.features)
    return proc_easyadapt(fe.features, domain="tgt")


