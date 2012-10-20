#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/test_classifier.py
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
from classifier import *

@attr("make_trcases")
class TestCaseMaker:
    def setUp(self):
        self.verbcorpus_dir = "../sandbox/classify/out"
        self.verbset_path = "../sandbox/classify/verbset_tiny.pkl2"
        self.model_dir = "../sandbox/classify/models"
        self.npy_dir = "../sandbox/classify/npy"


    def test_maketrcases(self):
        CM = CaseMaker(self.verbcorpus_dir, self.verbset_path, self.model_dir, self.npy_dir)
        CM.make_fvectors()
        raise Exception


@attr("bolt")
class TestBoltClassifier(object):
    def setUp(self):
        self.train3c = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/dataset.svmlight")
        self.test3c = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/dataset.svmlight")
        self.test3cone = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/testset.svmlight")
        self.correct = self.test3c.labels

    def test3classSGD(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=len(self.train3c.classes), biasterm = False)
        sgd = bolt.SGD(bolt.Hinge(), reg = 0.0001, epochs = 5)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        one_tc = array(self.test3c.instances[0])
        pred_c = [p for p in glm.predict(self.test3c.iterinstances(), confidence=True)]
        pred_one = [p for p in glm.predict(self.test3cone.iterinstances(), confidence=True)]
        # print sklearn.metrics.classification_report(self.correct, array(pred))
        # print pred_one
        # print pred_c
        try:
            # This worked... so the GLM model can be cPickled
            import cPickle as pickle
            pickle.dump(glm, open("../sandbox/classify/boltSGD_cPickled.pkl","wb"), -1)
            glm = pickle.load(open("../sandbox/classify/boltSGD_cPickled.pkl","rb"))
            pred2 = [p for p in glm.predict(self.test3c.iterinstances())]
            assert pred == pred2
        except:
            import Pickle as pickle
            pickle.dump(glm, open("../sandbox/classify/boltSGD_PyPickled.pkl","wb"), -1)
            glm = pickle.load(open("../sandbox/classify/boltSGD_PyPickled.pkl","rb"))
            pred2 = [p for p in glm.predict(self.test3c.iterinstances())]
            assert pred == pred2
    

    def test_actual_train_predict(self):
        BC = BoltClassifier()
        BC.read_traincases("../sandbox/classify/npy/have/dataset.svmlight")
        BC.train(model="sgd")
        pred = BC.predict(testset_path="../sandbox/classify/npy/have/testset.svmlight")
        print pred
        pred = BC.predict(testset_path="../sandbox/classify/npy/have/dataset.svmlight")
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception


@attr("bolt_pre")
class TestBoltClassifier1(object):
    def setUp(self):
        self.train3c = bolt.io.MemoryDataset.load("../sandbox/classify/train3c.svml")
        self.test3c = bolt.io.MemoryDataset.load("../sandbox/classify/train3c.svml")
        self.correct = self.test3c.labels

    def test3classSGD(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        sgd = bolt.SGD(bolt.Hinge(), reg = 0.0001, epochs = 50)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception

    def test3classPEGASOS(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        sgd = bolt.PEGASOS(reg = 0.0001, epochs = 50)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception

    def test3classAP(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        ap = bolt.AveragedPerceptron(epochs = 50)
        ap.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, array(pred))
        raise Exception

