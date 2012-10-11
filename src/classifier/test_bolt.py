#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/classifier/test_bolt.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"
from nose.plugins.attrib import attr
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from pprint import pformat
try: 
    import bolt
except:
    raise ImportError
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
from svmlight_loader import *
from numpy import array

class TestBoltClassifier(object):
    def setUp(self):
        self.train3c = bolt.io.MemoryDataset.load("../../sandbox/classify/train3c.svml")
        self.test3c = bolt.io.MemoryDataset.load("../../sandbox/classify/train3c.svml")
        self.correct = self.test3c.labels

    def test3classSGD(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        sgd = bolt.SGD(bolt.ModifiedHuber(), reg = 0.0001, epochs = 20)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception

    def test3classPEGASOS(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        sgd = bolt.PEGASOS(reg = 0.0001, epochs = 20)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception