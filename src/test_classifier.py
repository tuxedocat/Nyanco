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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction import DictVectorizer 
from sklearn.linear_model import SGDClassifier
from time import time
import numpy as np
from classifier import *
from tool.sparse_matrices import *

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

@attr("make_trcases_huge")
class TestCaseMaker_huge:
    def setUp(self):
        self.verbcorpus_dir = "../sandbox/classify/out"
        self.verbset_path = "../sandbox/classify/verbset_111_20.pkl2"
        self.model_dir = "../sandbox/classify/models"
        self.npy_dir = "../sandbox/classify/datasets"


    def test_maketrcases(self):
        CM = CaseMaker(self.verbcorpus_dir, self.verbset_path, self.model_dir, self.npy_dir)
        CM.make_fvectors()


@attr("bolt")
class TestBoltClassifier(object):
    def setUp(self):
        pass

    def test3classSGD(self):
        self.train3c = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/dataset.svmlight")
        self.test3c = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/dataset.svmlight")
        self.test3cone = bolt.io.MemoryDataset.load("../sandbox/classify/npy/have/testset.svmlight")
        self.correct = self.test3c.labels
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=len(self.train3c.classes), biasterm = False)
        sgd = bolt.SGD(bolt.Hinge(), reg = 0.0001, epochs = 5)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        one_tc = np.array(self.test3c.instances[0])
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
    
    @attr("bolt_actual")
    def test_actual_train_predict(self):
        BC = BoltClassifier()
        BC.read_traincases("../sandbox/classify/tiny/datasets/have/dataset.svmlight")
        BC.train(model="sgd")
        pred = BC.predict(testset_path="../sandbox/classify/tiny/datasets/have/dataset.svmlight")
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
        print sklearn.metrics.classification_report(self.correct, np.array(pred))
        raise Exception

    def test3classPEGASOS(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        sgd = bolt.PEGASOS(reg = 0.0001, epochs = 50)
        ova = bolt.OVA(sgd)
        ova.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, np.array(pred))
        raise Exception

    def test3classAP(self):
        glm = bolt.GeneralizedLinearModel(m=self.train3c.dim, k=3, biasterm = False)
        ap = bolt.AveragedPerceptron(epochs = 50)
        ap.train(glm, self.train3c, verbose=1, shuffle=True)
        pred = [p for p in glm.predict(self.test3c.iterinstances())]
        print sklearn.metrics.classification_report(self.correct, np.array(pred))
        raise Exception


@attr("sklearn")
class TestSklearnClassifier(object):
    def setUp(self):
        self.vec = DictVectorizer()
        X = self.vec.fit_transform([{"cat":1, "dog":0.4, "katze":0}, {"cat":0.1, "dog":0.5, "katze":1}])
        save_sparse_matrix("../sandbox/npy/X", X)
        self.X = load_sparse_matrix("../sandbox/npy/X.npz")
        self.Y = np.array([0, 1])
        Xm = self.vec.fit_transform([{"cat":1, "dog":0.4, "katze":0}, {"cat":0.1, "dog":0.5, "katze":1},
                                    {"cat":0.1, "dog":0.5, "katze":0.1}, {"cat":0.1, "dog":0, "katze":1}])
        self.Ym = np.array([0, 1, 2, 3])
        save_sparse_matrix("../sandbox/npy/Xm", Xm)
        self.Xm = load_sparse_matrix("../sandbox/npy/Xm.npz")

    def test_sklearnSGD(self):
        clf = SGDClassifier(loss = "hinge", penalty="l1")
        clf.fit(self.X, self.Y)
        pred = clf.predict(self.X)
        np.testing.assert_array_equal(self.Y, pred)

    def test_sklearnSGD_MC(self):
        clf = SGDClassifier(loss = "hinge", penalty="l1")
        ovr = OneVsRestClassifier(clf).fit(self.Xm, self.Ym)
        pred = ovr.predict(self.Xm)
        np.testing.assert_array_equal(self.Ym, pred)

    def test_actualSGD(self):
        dspath = "../sandbox/classify/tiny_sgd/have"
        outpath = dspath
        type = "sgd"
        opts = {"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}
        train_sklearn_classifier(dataset_dir=dspath, output_path=outpath, modeltype=type, cls_option=opts)
        modelpath = os.path.join(dspath, "model_sgd.pkl2")
    @attr("sklearn_p")
    def test_actualSGD_p(self):
        dspath = "../sandbox/classify/tiny_sgd/have"
        outpath = dspath
        type = "sgd"
        opts = {"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}
        train_sklearn_classifier(dataset_dir=dspath, output_path=outpath, modeltype=type, cls_option=opts)
        modelpath = os.path.join(dspath, "model_sgd.pkl2")