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
from copy import deepcopy

# Currently, assuming bolt online classifier toolkit as sgd/pegasos classifier
# and scikit-learn as utilities and for svm models
try: 
    import bolt
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn import preprocessing
    # from sklearn.datasets.svmlight_format import *
    from svmlight_loader import *
    import numpy as np
    import scipy as sp
except:
    raise ImportError
from feature_extractor import FeatureExtractor
from tool.sparse_matrices import *


class CaseMaker(object):
    def __init__(self, verbcorpus_dir=None, verbset_path=None, dataset_dir=None, restart_from=None, f_types=None):
        if not verbcorpus_dir and verbset_path and model_dir and dataset_dir:
            print "CaseMaker: Invalid data path(s)... aborted."
            raise TypeError
        else:
            pass
        self.corpusdir = verbcorpus_dir
        if not os.path.exists(os.path.abspath(dataset_dir)):
            os.makedirs(os.path.abspath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.verbset_path = verbset_path
        self.verbsets = pickle.load(open(verbset_path,"rb"))
        self.verbs = self.verbsets.keys()
        vcorpus_filenames = glob.glob(os.path.join(self.corpusdir, "*.pkl2"))
        v_names = [os.path.basename(path).split(".")[0] for path in vcorpus_filenames]
        self.vcorpus_filedic = {vn : fn for (vn, fn) in zip(v_names, vcorpus_filenames)}
        self.nullfeature = {"NULL":1}
        self.featuretypes = f_types # list like object is expected
        if restart_from:
            try:
                p_idx = self.verbs.index(restart_from)
                print "CaseMaker: restart from verb '%s' in #%d of list"%(restart_from, p_idx)
                self.verbs = self.verbs[p_idx:]
                old_vs = {v : vs for (v, vs) in self.verbsets.iteritems()}
                self.verbsets = {}
                for vn in self.verbs:
                    self.verbsets[vn] = old_vs[vn]
                print pformat(self.verbs)
                print pformat(self.verbsets)
            except Exception, e:
                print e


    def _is_validXY(self, X=[], Y=[]):
        try:
            if X.shape[0] == Y.shape[0]:
                return True
            else:
                return False
        except:
            return False


    def _read_verbcorpus(self, vset=None):
        _corpusdict = defaultdict(list)
        for v in [t[0] for t in vset]:
            print "CaseMaker make_fvectors: working on verb '%s'"%v
            try:
                with open(self.vcorpus_filedic[v], "rb") as vcf:
                    _corpusdict[v] = pickle.load(vcf)
            except:
                _corpusdict[v] = [[]]
        return _corpusdict


    def _get_features(self, v="", v_corpus=None, cls2id=None):
        _flist = []
        _labellist_int = []
        _labellist_str = []
        _labelid = cls2id[v]
        if v_corpus:
            for sid, s in enumerate(v_corpus):
                try:
                    fe = FeatureExtractor(s, verb=v)
                    if "ngram" in self.featuretypes:
                        fe.ngrams(n=5)
                    if "dep" in self.featuretypes:
                        fe.dependency()
                    if "srl" in self.featuretypes:
                        fe.srl()
                    if "ne" in self.featuretypes:
                        fe.ne()
                    if "errorprob" in self.featuretypes:
                        pass
                    _flist.append(fe.features)
                    _labellist_int.append(_labelid)
                    _labellist_str.append(v)
                except ValueError:
                    logging.debug(pformat("CaseMaker feature extraction: couldn't find the verb"))
                    pass
        else:
            _flist.append(self.nullfeature)
            _labellist_int.append(_labelid)
            _labellist_str.append(v)
        return _flist, _labellist_str, _labellist_int


    def make_fvectors(self):
        """
        Create feature vectors for given datasets, save dataset as numpy npz files
        """
        for setname, vset in self.verbsets.iteritems(): # setname is str, vset is list of tuples e.g. ("get", 25)
            if vset:
                print "CaseMaker make_fvectors: working on set '%s'"%setname
                vectorizer = DictVectorizer(sparse=True)
                _classname2id = {vt[0]: id for id, vt in enumerate(vset)}
                _casedict = defaultdict(list)
                _casedict["label2id"] = _classname2id
                _corpusdict = self._read_verbcorpus(vset)

                # Get feature vector for each sentence 
                for v, v_corpus in _corpusdict.iteritems():
                    _flist, _labellist_str, _labellist_int = self._get_features(v, v_corpus, _classname2id)
                    _casedict["X_str"] += _flist
                    _casedict["Y_str"] += _labellist_str
                    _casedict["Y"] += _labellist_int
                fvectors_str = _casedict["X_str"]

                try:
                    print "CaseMaker make_fvectors: Transforming string ID feature vectors into sparse matrix"
                    X = vectorizer.fit_transform(fvectors_str)
                    Y = np.array(_casedict["Y"])
                    if not self._is_validXY(X, Y):
                        raise UnboundLocalError
                except UnboundLocalError, e:
                    print "CaseMaker make_fvectors: seems feature vector for the set %s is empty..."%setname
                    print pformat(e)
                    print fvectors_str
                    # X = np.array([])
                    X = vectorizer.fit_transform(self.nullfeature)
                    Y = np.array([0.])
                dir_n = os.path.join(self.dataset_dir, setname)
                if not os.path.exists(dir_n):
                    os.makedirs(dir_n)
                fn_x = os.path.join(dir_n, "X")
                fn_y = os.path.join(dir_n, "Y")
                try:
                    save_sparse_matrix(fn_x, X)
                    np.save(fn_y, Y)
                except:
                    print "CaseMaker make_fvectors: Error occurred while saving npy, npz models"
                    raise
                fn_cdic = os.path.join(dir_n, "casedict.pkl2")
                fn_fmap = os.path.join(dir_n, "featuremap.pkl2")
                fn_label2id = os.path.join(dir_n, "label2id.pkl2")
                self.save_svmlight_file(dir_n, X, Y)
                with open(fn_fmap, "wb") as f:
                    pickle.dump(vectorizer, f, -1)
                with open(fn_label2id, "wb") as f:
                    pickle.dump(_casedict["label2id"], f, -1)
                with open(fn_cdic, "wb") as pf:
                    cdic = {"setname":setname}
                    cdic["X_str"] = _casedict["X_str"]; cdic["Y_str"] = _casedict["Y_str"]
                    cdic["label2id"] = _casedict["label2id"]
                    cdic["featuremap"] = vectorizer
                    pickle.dump(cdic, pf, -1)
            print "CaseMaker make_fvectors: successfully done."
        else:
            with open(self.verbset_path+"_reduced","wb") as f:
                verbs2 = [v for v in self.verbs if v != setname]
                vs2 = {sn : vs for (sn, vs) in self.verbsets.iteritems() if sn != setname}
                reduced = {"verbs": verbs2, "verbset": vs2}
                pickle.dump(reduced, f)
            dir_n = os.path.join(self.dataset_dir, setname)
            print "CaseMaker make_fvectors NOTIFICATION: Verbset is modified since there is null verbset"


    def save_svmlight_file(self, dir_n=None, X=None, Y=None):
        fn = os.path.join(dir_n, "dataset.svmlight")
        with open(fn, "wb") as f:
            dump_svmlight_file(X.tocsr(), Y, f)
            print "CaseMaker make_fvectors: Saving examples as SVMlight format..."


    # def save_svmlight_file(self, dir_n=None):
    #     fn = os.path.join(dir_n, "dataset.svmlight")
    #     fn_cdic = os.path.join(dir_n, "casedict.pkl2")
    #     fn_fmap = os.path.join(dir_n, "featuremap.pkl2")
    #     fn_label2id = os.path.join(dir_n, "label2id.pkl2")
    #     with open(fn+"temp", "wb") as f:
    #         print "CaseMaker make_fvectors: Saving examples as SVMlight format..."
    #         dump_svmlight_file(X, Y, f, comment=None)
    #     with open(fn+"temp", "rb") as f:
    #         cleaned = f.readlines()[2:]
    #     with open(fn, "wb") as f:
    #         f.writelines(cleaned)
    #         os.remove(fn+"temp")
    #     with open(fn_fmap, "wb") as f:
    #         pickle.dump(vectorizer, f, -1)
    #     with open(fn_label2id, "wb") as f:
    #         pickle.dump(_casedict["label2id"], f, -1)
    #     with open(fn_cdic, "wb") as pf:
    #         cdic = {"setname":setname}
    #         cdic["X_str"] = _casedict["X_str"]; cdic["Y_str"] = _casedict["Y_str"]
    #         cdic["label2id"] = _casedict["label2id"]
    #         cdic["featuremap"] = vectorizer
    #         pickle.dump(cdic, pf, -1)


def make_fvectors(verbcorpus_dir=None, verbset_path=None, dataset_dir=None, restart_from=None, f_types=None):
    CM = CaseMaker(verbcorpus_dir=verbcorpus_dir, verbset_path=verbset_path, dataset_dir=dataset_dir, restart_from=restart_from, f_types=f_types)
    CM.make_fvectors()


#----------------------------------------------------------------------------------------------------
class BaseClassifier(object):
    def __init__(self, mtype=None, opts=None):
        self.glm = None
        self.opts = opts
        self.multicpu = 1
        self.modeltype = mtype

    def setopts(self): 
        if "lambda" in self.opts:
            self.lmd = self.opts["lambda"]
        if "epochs" in self.opts:
            self.epochs = self.opts["epochs"]
        if "loss" in self.opts:
            self.loss = self.opts["loss"]
        if "reg" in self.opts:
            self.reg = self.opts["reg"]
        if "alpha" in self.opts:
            self.alpha = self.opts["alpha"]
        if "multicpu" in self.opts:
            self.multicpu = -1
        if "shuffle" in self.opts:
            self.shuffle = True

    def save_model(self, output_path=None):
        try:
            with open(output_path, "wb") as f:
                pickle.dump(self.glm, f, -1)
        except Exception, e:
            print pformat(e)
            with open(output_path, "wb") as f: 
                pass

    def load_model(self, model_path=None):
        try:
            with open(model_path, "rb") as f:
                self.glm = pickle.load(f)
        except:
            raise

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class SklearnClassifier(BaseClassifier):
    def load_dataset(self, dataset_path=None):
        try:
            self.X = load_sparse_matrix(os.path.join(dataset_path, "X.npz"))
            self.Y = np.load(os.path.join(dataset_path, "Y.npy"))
        except IOError:
            print "Seems model files (X.npz, Y.npy) are not found..."

    def trainSGD(self):
        sgd = SGDClassifier(loss=self.loss, penalty=self.reg, alpha=self.alpha, n_iter=self.epochs,
                            shuffle=True, n_jobs=self.multicpu)
        print "Classifier (sklearn SGD): training the model"
        self.glm = OneVsRestClassifier(sgd).fit(self.X, self.Y)
        print "Classifier (sklearn SGD): Done."

    def predict(self, testset_path=None, X=None, Y=None):
        fn_x = os.path.join(testset_path, "X.npz")
        fn_y = os.path.join(testset_path, "Y.npy")
        Xtest = load_sparse_matrix(fn_x)
        Ytest = np.load(fn_y)
        pred = self.glm.predict(Xtest)
        return pred

    def predict_prob(self, testset_path=None, X=None, Y=None):
        """
        Parameters
        ----------
        testset_path: path prefix for testset .npz and .npy file

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        Y : numpy array of shape [n_samples]
            Multi-class targets.

        Returns
        -------
        pred_p: {array-like}, shape = {n_samples, n_classes}
        """
        if hasattr(self.glm, "predict_prob"):
            fn_x = os.path.join(testset_path, "X.npz")
            fn_y = os.path.join(testset_path, "Y.npy")
            Xtest = load_sparse_matrix(fn_x)
            Ytest = np.load(fn_y)
            pred_p = self.glm.predict_prob(Xtest)
            return pred_p
        else:
            raise NotImplementedError


def train_sklearn_classifier(dataset_dir="", output_path="", modeltype="sgd", 
                            cls_option={"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}):
    classifier = SklearnClassifier(mtype=modeltype, opts=cls_option)
    classifier.setopts()
    classifier.load_dataset(dataset_dir)
    if modeltype == "sgd":
        classifier.trainSGD()
    modelfilename = os.path.join(output_path, "model_%s.pkl2"%modeltype)
    classifier.save_model(modelfilename)
    # _selftest_sk(modelfilename, dataset_dir)


def _selftest_sk(modelpath="", dspath=""):
    X = load_sparse_matrix(os.path.join(dspath, "X.npz"))
    glm = pickle.load(open(modelpath, "rb"))
    Y = np.load(os.path.join(dspath, "Y.npy"))
    pred = glm.predict(X)
    from sklearn.metrics import classification_report
    print classification_report(Y, pred)


def train_sklearn_classifier_batch(dataset_dir="", modeltype="sgd", verbset_path="", selftest=False, 
                                cls_option={"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}):
    vs_file = pickle.load(open(verbset_path, "rb"))
    verbs = vs_file.keys()
    verbsets = deepcopy(vs_file)
    set_names = [os.path.join(dataset_dir, v) for v in verbs]
    for idd, dir in enumerate(set_names):
        dspath = os.path.join(dir)
        print "Batch trainer (sklearn %s):started\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        train_sklearn_classifier(dataset_dir=dspath, output_path=dir, modeltype=modeltype, cls_option=cls_option)
        print "Batch trainer (sklearn %s):done!\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        if selftest:
            print "Batch trainer selftest..."
            _selftest_sk(modelfilename, dspath)
            print "Batch trainer selftest... done!"



class BoltClassifier(BaseClassifier):
    def __init__(self):
        self.models = ["sgd", "pegasos", "ap"]


    def read_traincases(self, dataset_path=""):
        try:
            self.training_dataset = bolt.io.MemoryDataset.load(dataset_path)
        except Exception, e:
            print pformat(e)


    def train(self, model="sgd", opt={"loss":"hinge", "epochs":10, "lambda":0.0001, "reg":"L2"}):
        try:
            if "lambda" in opt:
                l = opt["lambda"]
            if "epochs" in opt:
                epochs = opt["epochs"]
            if "loss" in opt:
                loss = opt["loss"]
                if loss == "huber":
                    loss = bolt.ModifiedHuber()
                elif loss == "hinge":
                    loss = bolt.Hinge()
            if "reg" in opt:
                reg = opt["reg"]
                if reg == "L1":
                    reg = 1
                elif reg == "L2":
                    reg = 2
                else:
                    reg =3
            self.glm = bolt.GeneralizedLinearModel(m=self.training_dataset.dim, 
                                                   k=len(self.training_dataset.classes))
            if model == "sgd":
                trainer = bolt.SGD(loss=loss, reg=l, epochs=epochs, norm=reg)
            elif model == "pegasos":
                trainer = bolt.PEGASOS(reg=l, epochs=epochs)
            elif model == "ap":
                trainer = bolt.AveragedPerceptron(epochs=epochs)
            else:
                raise NotImplementedError
            if model == "ap":
                trainer.train(self.glm, self.training_dataset, verbose=0, shuffle=True)
            else:
                ova = bolt.OVA(trainer)
                ova.train(self.glm, self.training_dataset, verbose=1, shuffle=True)
        except Exception, e:
            print pformat(e)


    def save_model(self, output_path=None):
        try:
            with open(output_path, "wb") as f:
                pickle.dump(self.glm, f, -1)
        except Exception, e:
            print pformat(e)
            with open(output_path, "wb") as f: 
                pass

    def load_model(self, model_path=None):
        try:
            with open(model_path, "rb") as f:
                self.glm = pickle.load(f)
        except:
            raise

    def predict(self, testset_path=None, testset_array=None):
        if testset_array:
            raise NotImplementedError
        elif testset_path:
            testset = bolt.io.MemoryDataset.load(testset_path)
        pred = [p for p in self.glm.predict(testset.iterinstances())]
        return pred

def train_boltclassifier(dataset_path="", output_path="", modeltype="sgd", 
                            cls_option={"loss":"hinge", "epochs":10, "lambda":0.0001, "reg":"L2"}):
    classifier = BoltClassifier()
    classifier.read_traincases(dataset_path)
    classifier.train(model=modeltype, opt=cls_option)
    classifier.save_model(output_path)

def _selftest(modelpath="", dspath=""):
    boltdataset = bolt.io.MemoryDataset.load(dspath)
    glm = pickle.load(open(modelpath, "rb"))
    correct = boltdataset.labels
    testset = boltdataset.instances
    pred = [p for p in glm.predict(testset)]
    predc = [p for p in glm.predict(testset, confidence=True)]
    print predc
    from sklearn.metrics import classification_report
    print classification_report(correct, np.array(pred))

def train_boltclassifier_batch(dataset_dir="", modeltype="sgd", verbset_path="", selftest=True, 
                                cls_option={"loss":"hinge", "epochs":10, "lambda":0.0001, "reg":"L2"}):
    vs_file = pickle.load(open(verbset_path, "rb"))
    verbs = vs_file.keys()
    verbsets = deepcopy(vs_file)
    set_names = [os.path.join(dataset_dir, v) for v in verbs]
    for idd, dir in enumerate(set_names):
        modelfilename = os.path.join(dir, "model_%s.pkl2"%modeltype)
        dspath = os.path.join(dir, "dataset.svmlight")
        print "Batch trainer (bolt %s):started\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        train_boltclassifier(dataset_path=dspath, output_path=modelfilename, modeltype=modeltype)
        print "Batch trainer (bolt %s):done!\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
        if selftest:
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
    description = """python classifier.py -M prepare -c ../sandbox/classify/tiny/out -v ../sandbox/classify/verbset_tiny.pkl2 -d ../sandbox/classify/tiny/datasets --restart_from get\n
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
    ap.add_argument("-r", '--restart_from', action="store",
                    help="input previous stop point (e.g. want)")
    ap.add_argument('--selftest', action="store_true",
                    help="If selftest of classifier is needed")
    args = ap.parse_args()

    if (args.Mode=="prepare"):
        make_fvectors(args.verbcorpus_path, args.verbset_path, args.dataset_dir, args.restart_from)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    elif (args.Mode=="train_save"):
        train_boltclassifier_batch(args.dataset_dir, args.modeltype, args.verbset_path, args.selftest)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    else:
        ap.print_help()
    quit()