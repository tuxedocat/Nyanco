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
    from sklearn.datasets.svmlight_format import *
    import numpy as np
except:
    raise ImportError
from feature_extractor import SimpleFeatureExtractor


class CaseMaker(object):
    def __init__(self, verbcorpus_dir="", verbset_path="", dataset_dir=""):
        if not verbcorpus_dir and verbset_path and model_dir and dataset_dir:
            raise TypeError
        else:
            print "CaseMaker: successfully imported bolt and sklearn"
        self.corpusdir = verbcorpus_dir
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset_dir = dataset_dir
        verbset_load = pickle.load(open(verbset_path,"rb"))
        self.verbs = verbset_load["verbs"]
        self.verbsets = verbset_load["verbset"]
        vcorpus_filenames = glob.glob(os.path.join(self.corpusdir, "*.pkl2"))
        v_names = [os.path.basename(path).split(".")[0] for path in vcorpus_filenames]
        self.vcorpus_filedic = {vn : fn for (vn, fn) in zip(v_names, vcorpus_filenames)}
        self.nullfeature = {"NULL":1}


    def make_fvectors(self):
        """
        Create feature vectors for given datasets, for classifiers as SVM^light format
        using feature_extraction's classes
        """
        for setname, vset in self.verbsets.iteritems(): # setname is str, vset is list
            print "CaseMaker make_fvectors: working on set '%s'"%setname
            vectorizer = DictVectorizer(sparse=True)
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
                if v_corpus:
                    for sid, s in enumerate(v_corpus):
                        fe = SimpleFeatureExtractor(s, verb=v)
                        fe.ngrams(n=7)
                        # some other features!
                        # then finally...
                        _flist.append(fe.features)
                        _labellist_int.append(_labelid)
                        _labellist_str.append(v)
                else:
                        _flist.append(self.nullfeature)
                        _labellist_int.append(_labelid)
                        _labellist_str.append(v)
                _casedict["X_str"] += _flist
                _casedict["Y_str"] += _labellist_str
                _casedict["Y"] += _labellist_int
            fvectors_str = _casedict["X_str"]
            try:
                print "CaseMaker make_fvectors: Transforming string ID feature vectors"
                X = vectorizer.fit_transform(fvectors_str)
                Y = np.array(_casedict["Y"])
            except UnboundLocalError, e:
                print "CaseMaker make_fvectors: seems feature vector for the set %s is empty..."%setname
                print pformat(e)
                print fvectors_str
                X = np.array([])
                Y = np.array([])
            dir_n = os.path.join(self.dataset_dir, setname)
            if not os.path.exists(dir_n):
                os.makedirs(dir_n)
            fn = os.path.join(dir_n, "dataset.svmlight")
            fn_cdic = os.path.join(dir_n, "casedict.pkl2")
            with open(fn+"temp", "wb") as f:
                print "CaseMaker make_fvectors: Saving examples as SVMlight format..."
                dump_svmlight_file(X, Y, f, comment=None)
            with open(fn+"temp", "rb") as f:
                cleaned = f.readlines()[2:]
            with open(fn, "wb") as f:
                f.writelines(cleaned)
                os.remove(fn+"temp")

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


    def read_traincases(self, dataset_path=""):
        # try:
        self.training_dataset = bolt.io.MemoryDataset.load(dataset_path)
        # except Exception, e:
        #     print pformat(e)


    def train(self, model="sgd", params={"reg":0.0001, "epochs": 30}):
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
            trainer.train(self.glm, self.training_dataset, verbose=0, shuffle=True)
        else:
            ova = bolt.OVA(trainer)
            ova.train(self.glm, self.training_dataset, verbose=1, shuffle=True)


    def save_model(self, output_path=""):
        try:
            with open(output_path, "wb") as f:
                pickle.dump(self.glm, f, -1)
        except:
            raise

    def load_model(self, model_path=""):
        try:
            with open(model_path, "rb") as f:
                self.glm = pickle.load(f)
        except:
            raise

    def predict(self, testset_path="", testset_array=[]):
        if testset_array:
            raise NotImplementedError
        elif testset_path:
            testset = bolt.io.MemoryDataset.load(testset_path)
        pred = [p for p in self.glm.predict(testset.iterinstances())]
        return pred


def make_fvectors(verbcorpus_dir, verbset_path, dataset_dir):
    CM = CaseMaker(verbcorpus_dir, verbset_path, dataset_dir)
    CM.make_fvectors()

def train_boltclassifier(dataset_path="", output_path="", modeltype="sgd"):
    default = {"reg":0.0001, "epochs": 30}
    classifier = BoltClassifier()
    classifier.read_traincases(dataset_path)
    classifier.train(model=modeltype, params=default)
    classifier.save_model(output_path)

def _selftest(modelpath="", dspath=""):
    boltdataset = bolt.io.MemoryDataset.load(dspath)
    glm = pickle.load(open(modelpath, "rb"))
    correct = boltdataset.labels
    testset = boltdataset.instances
    pred = [p for p in glm.predict(testset)]
    from sklearn.metrics import classification_report
    print classification_report(correct, np.array(pred))

def train_boltclassifier_batch(dataset_dir="", modeltype="sgd"):
    set_names = glob.glob(os.path.join(dataset_dir, "*"))
    v_names = [os.path.basename(path) for path in set_names]
    fndic = {vn : dn for (vn, dn) in zip(v_names, set_names)}
    for idd, dir in enumerate(set_names):
        modelfilename = os.path.join(dir, "model_%s.pkl2"%modeltype)
        dspath = os.path.join(dir, "dataset.svmlight")
        print "Batch trainer (bolt %s):started\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        train_boltclassifier(dataset_path=dspath, output_path=modelfilename, modeltype=modeltype)
        print "Batch trainer (bolt %s):done!\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        print "Batch trainer selftest..."
        _selftest(modelfilename, dspath)
        print "Batch trainer selftest... done!"
if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description = """python classifier.py -M prepare -c ../sandbox/classify/tiny/out -v ../sandbox/classify/verbset_tiny.pkl2 -d ../sandbox/classify/tiny/datasets\n
python classifier.py -M train_save -d ../sandbox/classify/tiny/datasets -m sgd
"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--verbcorpus_path", action="store", 
                    help="path to pickled corpus files")
    ap.add_argument("-o", '--model_save_dir', action="store",
                    help="path to trained classifier model directory")
    ap.add_argument("-m", '--modeltype', action="store",
                    help="sgd | pegasos | ap   (default: sgd)")
    ap.add_argument("-v", '--verbset_path', action="store",
                    help="path of verbset pickle file")
    ap.add_argument("-d", '--dataset_dir', action="store",
                    help="model store for .svmlight examples")
    ap.add_argument("-M", '--Mode', action="store",
                    help="set 'prepare' for making f_vectors, and 'train_save' for training and saving the models")
    args = ap.parse_args()

    if (args.Mode=="prepare"):
        make_fvectors(args.verbcorpus_path, args.verbset_path, args.dataset_dir)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    elif (args.Mode=="train_save"):
        train_boltclassifier_batch(args.dataset_dir, args.modeltype)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    else:
        ap.print_help()
    quit()