#! /usr/bin/env python
# coding: utf-8
'''
nyanco/src/classifer.py
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
from collections import defaultdict
from pprint import pformat
from time import time
import glob
# Currently, assuming bolt online classifier toolkit as sgd/pegasos classifier
# and scikit-learn as utilities and for svm models
try: 
    import bolt
    from sklearn.feature_extraction import DictVectorizer
    from sklearn import preprocessing
    from svmlight_loader import *
    import numpy as np
except:
    raise ImportError
from feature_extractor import SimpleFeatureExtractor


class CaseMaker(object):
    def __init__(self, verbcorpus_dir="", verbset_path="", model_dir="", npy_dir=""):
        if not verbcorpus_dir and verbset_path and model_dir and npy_dir:
            raise TypeError
        else:
            print "CaseMaker: successfully imported bolt and sklearn"
        self.corpusdir = verbcorpus_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        self.model_dir = model_dir
        self.npy_dir = npy_dir
        verbset_load = pickle.load(open(verbset_path,"rb"))
        self.verbs = verbset_load["verbs"]
        self.verbsets = verbset_load["verbset"]
        vcorpus_filenames = glob.glob(os.path.join(self.corpusdir, "*.pkl2"))
        v_names = [os.path.basename(path).split(".")[0] for path in vcorpus_filenames]
        self.vcorpus_filedic = {vn : fn for (vn, fn) in zip(v_names, vcorpus_filenames)}


    def make_fvectors(self):
        """
        Create feature vectors for given datasets, for classifiers as SVM^light format
        using feature_extraction's classes
        """
        for setname, vset in self.verbsets.iteritems(): # setname is str, vset is list
            print "CaseMaker make_fvectors: working on set '%s'"%setname
            vectorizer = DictVectorizer(sparse=True)
            # label_encoder = preprocessing.LabelEncoder()
            _classname2id = {vt[0]: id for id, vt in enumerate(vset)}
            _corpusdict = {}
            _casedict = defaultdict(list)
            _casedict["label2id"] = _classname2id
            for v in [t[0] for t in vset]:
                with open(self.vcorpus_filedic[v], "rb") as vcf:
                    _corpusdict[v] = pickle.load(vcf)
            for v, v_corpus in _corpusdict.iteritems():
                _flist = []
                _labellist_int = []
                _labellist_str = []
                _labelid = _classname2id[v]
                for sid, s in enumerate(v_corpus):
                    fe = SimpleFeatureExtractor(s, verb=v)
                    fe.ngrams(n=7)
                    # some other features!
                    # then finally...
                    _flist.append(fe.features)
                    _labellist_int.append(_labelid)
                    _labellist_str.append(v)
                _casedict["X_str"] += _flist
                _casedict["Y_str"] += _labellist_str
                _casedict["Y"] += _labellist_int
            fvectors_str = _casedict["X_str"]
            try:
                X = vectorizer.fit_transform(fvectors_str).toarray()
                Y = np.array(_casedict["Y"])
                dim_X = X.shape[1]
            except UnboundLocalError, e:
                print "CaseMaker make_fvectors: seems feature vector for the set %s is empty..."%setname
                print pformat(e)
                print fvectors_str
                X = array([])
                Y = array([])
                dim_X = 0
            dir_n = os.path.join(self.npy_dir, setname)
            if not os.path.exists(dir_n):
                os.makedirs(dir_n)
            fn = os.path.join(dir_n, "dataset.svmlight")
            fn_cdic = os.path.join(dir_n, "casedict.pkl2")
            with open(fn, "wb") as f:
                # np.savez(f, instances=X, labels=Y, dim=dim_X)
                dump_svmlight_file(X, Y, f)
            with open(fn_cdic, "wb") as pf:
                cdic = {"setname":setname}
                cdic["X_str"] = _casedict["X_str"]; cdic["Y_str"] = _casedict["Y_str"]
                cdic["label2id"] = _casedict["label2id"]
                pickle.dump(cdic, pf, -1)
        print "CaseMaker make_fvectors: successfully done."

#----------------------------------------------------------------------------------------------------
class Classifier(object):
    def __init__(self):
        pass

    def read(self, x, y):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class BoltClassifier(Classifier):
    def __init__(self):
        self.models = ["sgd", "pegasos", "ap"]
        pass

    def read_traincases(self, dataset_path=""):
        try:
            self.training_dataset = bolt.io.MemoryDataset.load(dataset_path)
        except Exception, e:
            print pformat(e)

    def train(self, model="sgd", params={"reg":0.0001, "epochs": 20}):
        if "reg" in params:
            reg = params["reg"]
        if "epochs" in params:
            epochs = params["epochs"]
        self.glm = bolt.GeneralizedLinearModel(m=self.training_dataset.dim, 
                                               k=len(self.training_dataset.classes))
        if model == "sgd":
            trainer = bolt.SGD(bolt.Hinge(), reg=reg, epochs=epochs)
        elif model == "pegasos":
            trainer = bolt.PEGASOS(reg=reg, epochs=epochs)
        elif model == "ap":
            trainer = bolt.AveragedPerceptron(epochs=epochs)
        else:
            raise NotImplementedError
        if model == "ap":
            ap.train(self.glm, self.training_dataset, verbose=1, shuffle=True)
        else:
            ova = bolt.OVA(trainer)
            ova.train(self.glm, self.training_dataset, verbose=1, shuffle=True)


    def save_model(self, output_path=""):
        try:
            with open(output_path, "wb") as f:
                pickle.dump(self.glm, f, -1)
        except:
            raise


    def predict(self, testset_path="", testset_array=[]):
        if testset_array:
            raise NotImplementedError
        elif testset_path:
            testset = bolt.io.MemoryDataset.load(testset_path)
        pred = [p for p in self.glm.predict(testset.iterinstances())]
        return pred


def make_fvectors():
    verbcorpus_dir = "../sandbox/classify/out"
    verbset_path = "../sandbox/classify/verbset_111_20.pkl2"
    model_dir = "../sandbox/classify/models"
    npy_dir = "../sandbox/classify/datasets"
    CM = CaseMaker(verbcorpus_dir, verbset_path, model_dir, npy_dir)
    CM.make_fvectors()

if __name__=="__main__":
    import time
    start_time = time.time()
    make_fvectors()
    print "done in %6.3f[sec.]"%(time.time()-start_time)
