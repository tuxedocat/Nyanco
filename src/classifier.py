#!/usr/bin/env python
#coding: utf-8
'''
nyanco/src/classifer.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import os
import errno
import sys
import traceback
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
import collections
from collections import defaultdict
from pprint import pformat
from time import time
import glob
from copy import deepcopy
from multiprocessing import Pool
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
                        FileTransferSpeed, FormatLabel, Percentage, \
                        ProgressBar, ReverseBar, RotatingMarker, \
                        SimpleProgress, Timer
# Currently, scikit-learn 0.13 git is primal toolkit for classifiers
# Alternatively Bolt online-learning toolkit is used
try: 
    import bolt
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multiclass import OutputCodeClassifier
    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.svm import NuSVC, SVC
    from sklearn import preprocessing
    # from sklearn.datasets.svmlight_format import *
    from svmlight_loader import *
    import numpy as np
    import scipy as sp
except:
    raise ImportError
from feature_extractor import FeatureExtractor, SentenceFeatures, proc_easyadapt
from tool.sparse_matrices import *
from tool.seq_chunker import chunk_gen



class CaseMaker(object):
    def __init__(self, verbcorpus_dir=None, verbset_path=None, dataset_dir=None, restart_from=None, f_types=None):
        if not verbcorpus_dir and verbset_path and f_types and dataset_dir:
            print "CaseMaker: Invalid data path(s)... aborted."
            raise TypeError
        else:
            pass
        self.corpusdir = verbcorpus_dir
        try:
            os.makedirs(os.path.abspath(dsdir))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
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
        self.numts = 30000


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
            # print "CaseMaker make_fvectors: working on verb '%s'"%v
            try:
                with open(self.vcorpus_filedic[v], "rb") as vcf:
                    _corpusdict[v] = pickle.load(vcf)[:self.numts]
            except:
                _corpusdict[v] = [[]]
        return _corpusdict


    def _get_features(self, v="", v_corpus=None, cls2id=None, domain="src"):
        _flist = []
        _labellist_int = []
        _labellist_str = []
        _labelid = cls2id[v]
        if v_corpus:
            for sid, s in enumerate(v_corpus):
                try:
                    fe = FeatureExtractor(s, verb=v)
                    if "chunk" in self.featuretypes:
                        fe.chunk()
                    if "3gram" in self.featuretypes:
                        fe.ngrams(n=3)
                    if "5gram" in self.featuretypes:
                        fe.ngrams(n=5)
                    if "7gram" in self.featuretypes:
                        fe.ngrams(n=7)
                    if "dep" in self.featuretypes:
                        fe.dependency()
                    if "srl" in self.featuretypes:
                        fe.srl()
                    if "ne" in self.featuretypes:
                        fe.ne()
                    if "errorprob" in self.featuretypes:
                        pass
                    if "topic" in self.featuretypes:
                        pass
                    augf = proc_easyadapt(fe.features, domain=domain)
                    _flist.append(augf)
                    _labellist_int.append(_labelid)
                    _labellist_str.append(v)
                except ValueError:
                    logging.debug(pformat("CaseMaker feature extraction: couldn't find the verb"))
                except:
                    print v
                    raise
        else:
            _flist.append(self.nullfeature)
            _labellist_int.append(_labelid)
            _labellist_str.append(v)
        return _flist, _labellist_str, _labellist_int

    def _get_features_tgt(self, v_corpus=None, cls2id=None, domain="tgt"):
        _flist = []
        _labellist_int = []
        _labellist_str = []
        for sid, sdic in enumerate(v_corpus):
            v = sdic["label_corr"]
            _labelid = cls2id[v]
            try:
                fe = SentenceFeatures(sdic["parsed_corr"], verb=v, v_idx=sdic["vidx_corr"])
                if "chunk" in self.featuretypes:
                    fe.chunk()
                if "3gram" in self.featuretypes:
                    fe.ngrams(n=3)
                if "5gram" in self.featuretypes:
                    fe.ngrams(n=5)
                if "7gram" in self.featuretypes:
                    fe.ngrams(n=7)
                if "dep" in self.featuretypes:
                    fe.dependency()
                if "srl" in self.featuretypes:
                    fe.srl()
                if "ne" in self.featuretypes:
                    fe.ne()
                if "errorprob" in self.featuretypes:
                    pass
                if "topic" in self.featuretypes:
                    pass
                augf = proc_easyadapt(fe.features, domain=domain)
                assert augf and _labelid and v
                _flist.append(augf)
                _labellist_int.append(_labelid)
                _labellist_str.append(v)
            except ValueError:
                logging.debug(pformat("CaseMaker feature extraction: couldn't find the verb"))
            except:
                print v
        # else:
            # _flist.append(self.nullfeature)
            # _labellist_int.append(_labelid)
            # _labellist_str.append(v)
        return _flist, _labellist_str, _labellist_int


    def make_fvectors(self):
        """
        Create feature vectors for given datasets, save dataset as numpy npz files
        """
        pbar = ProgressBar(widgets=[Percentage(),'(', SimpleProgress(), ')  ', Bar()], 
                           maxval=len(self.verbsets)).start()
        for _i, (setname, vset) in enumerate(self.verbsets.iteritems()): # setname is str, vset is list of tuples e.g. ("get", 25)
            if vset:
                # print "CaseMaker make_fvectors: working on set '%s'"%setname
                vectorizer = DictVectorizer(sparse=True)
                _classname2id = {vt[0]: id for id, vt in enumerate(vset)}
                _casedict = defaultdict(list)
                _casedict["label2id"] = _classname2id
                _corpusdict = self._read_verbcorpus(vset)

                # Get feature vector for each sentence 
                for v, v_corpus in _corpusdict.iteritems():
                    try:
                        _flist, _labellist_str, _labellist_int = self._get_features(v, v_corpus, _classname2id, "src")
                    except:
                        print v
                        raise
                    _casedict["X_str"] += _flist
                    _casedict["Y_str"] += _labellist_str
                    _casedict["Y"] += _labellist_int
                del(_corpusdict)

                if self.easyadapt:
                    _tgtc = self.tgtcorpus[setname]
                    try:
                        _flist, _labellist_str, _labellist_int = self._get_features_tgt(_tgtc, _classname2id, "tgt")
                    except:
                        # print _tgtc
                        # raise
                        pass
                    _casedict["X_str"] += _flist
                    _casedict["Y_str"] += _labellist_str
                    _casedict["Y"] += _labellist_int

                try:
                    X = vectorizer.fit_transform(_casedict["X_str"])
                    Y = np.array(_casedict["Y"])
                    if not self._is_validXY(X, Y):
                        raise UnboundLocalError
                except UnboundLocalError, e:
                    print "CaseMaker make_fvectors: seems feature vector for the set %s is empty..."%setname
                    print pformat(e)
                    X = vectorizer.fit_transform(self.nullfeature)
                    Y = np.array([0.])
                dir_n = os.path.join(self.dataset_dir, setname)
                self.save_npz_file(dir_n, X, Y)
                # self.save_svmlight_file(dir_n, X, Y)
                fn_cdic = os.path.join(dir_n, "casedict.pkl2")
                fn_fmap = os.path.join(dir_n, "featuremap.pkl2")
                fn_label2id = os.path.join(dir_n, "label2id.pkl2")
                with open(fn_fmap, "wb") as f:
                    pickle.dump(vectorizer, f, -1)
                with open(fn_label2id, "wb") as f:
                    pickle.dump(_casedict["label2id"], f, -1)
                del(_casedict)
                print "CaseMaker make_fvectors: successfully done for a confusion set '%s'"%setname
                pbar.update(_i+1)
            else:
                print "CaseMaker make_fvectors: NULL VERBSET is found (setname = %s)"%setname
                pbar.update(_i+1)
        pbar.finish()

    def save_npz_file(self, dir_n=None, X=None, Y=None):
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

    def save_svmlight_file(self, dir_n=None, X=None, Y=None):
        fn = os.path.join(dir_n, "dataset.svmlight")
        with open(fn, "wb") as f:
            dump_svmlight_file(X.tocsr(), Y, f)
            print "CaseMaker make_fvectors: Saving examples as SVMlight format..."

    # def save_subfiles(self, dir_n=None, vec=None, cdic=None, fmap=None, label2id=None):
        # fn_cdic = os.path.join(dir_n, "casedict.pkl2")
        # fn_fmap = os.path.join(dir_n, "featuremap.pkl2")
        # fn_label2id = os.path.join(dir_n, "label2id.pkl2")
        # with open(fn_fmap, "wb") as f:
            # pickle.dump(vec, f, -1)
        # with open(fn_label2id, "wb") as f:
            # pickle.dump(label2id, f, -1)
        # # with open(fn_cdic, "wb") as pf:
            # # cdic = {"setname":setname}
            # # cdic["X_str"] = _casedict["X_str"]; cdic["Y_str"] = _casedict["Y_str"]
            # # cdic["label2id"] = _casedict["label2id"]
            # # cdic["featuremap"] = vectorizer
            # # pickle.dump(cdic, pf, -1)
        # print "CaseMaker make_fvectors: successfully done for a confusion set '%s'"%setname
        # pbar.update(_i+1)
    # else:
        # print "CaseMaker make_fvectors: NULL VERBSET is found (setname = %s)"%setname
        # pbar.update(_i+1)



class ParallelCaseMaker(CaseMaker):

    def __init__(self, vcdir=None, vs={}, dsdir=None, f_types=None, instance_num=30000, 
                 easyadapt=False, tgtcorpus={}):
        if not vcdir and vs and dsdir and f_types:
            print "ParallelCaseMaker: Invalid data path(s)... aborted."
            raise TypeError
        else:
            pass
        self.corpusdir = vcdir
        try:
            os.makedirs(os.path.abspath(dsdir))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        self.dataset_dir = dsdir
        self.verbsets = vs 
        self.verbs = self.verbsets.keys()
        vcorpus_filenames = glob.glob(os.path.join(self.corpusdir, "*.pkl2"))
        v_names = [os.path.basename(path).split(".")[0] for path in vcorpus_filenames]
        self.vcorpus_filedic = {vn : fn for (vn, fn) in zip(v_names, vcorpus_filenames)}
        self.nullfeature = {"NULL":0}
        self.featuretypes = f_types # list like object is expected
        self.numts = instance_num
        self.easyadapt = easyadapt
        self.tgtcorpus = tgtcorpus


def make_fvectors(verbcorpus_dir=None, verbset_path=None, dataset_dir=None, 
                  f_types=None, pool_num=2, instance_num=30000, easyadapt=False, tgtcorpus_path=None):
    args = []
    argd = {}
    with open(verbset_path, "rb") as f:
        vs_full = pickle.load(f)
    if easyadapt:
        with open(tgtcorpus_path, "rb") as f:
            tgtc_full = pickle.load(f)
    else:
        tgtc_full = None
    # sep_keys = [wl for wl in chunk_gen(vs_full.keys(), (len(vs_full)/(pool_num*8))+1)]
    sep_keys = [wl for wl in chunk_gen(vs_full.keys(), 2)]
    vs_chunks = []
    tgtc_chunks = []
    for wl in sep_keys:
        vs_chunks.append({w:vs_full[w] for w in wl})
    if tgtc_full:
        for wl in sep_keys:
            tgtc_chunks.append({w:tgtc_full[w] for w in wl})
    else:
        tgtc_chunks = [None for item in vs_chunks]
    # print vs_chunks[:5]
    # print tgtc_chunks[:5]
    for vs, tgtc in zip(vs_chunks, tgtc_chunks):
        args.append({"vcdir":verbcorpus_dir, 
                     "dsdir":dataset_dir, 
                     "f_types":f_types, 
                     "vs":vs, 
                     "numts":instance_num,
                     "easyadapt":easyadapt,
                     "tgtcorpus":tgtc})
    print pformat(args)
    mp = Pool(processes=pool_num, maxtasksperchild=1)
    mp.map(_make_fvectors_p, args)
    mp.close()
    mp.join()


def _make_fvectors_p(argd):
    vcdir = argd["vcdir"]
    vs = argd["vs"]
    dsdir = argd["dsdir"]
    f_types = argd["f_types"]
    numts = argd["numts"]
    easyadapt = argd["easyadapt"]
    tgtcorpus = argd["tgtcorpus"]
    CMP = ParallelCaseMaker(vcdir=vcdir, 
                            vs=vs, 
                            dsdir=dsdir, 
                            f_types=f_types, 
                            instance_num=numts, 
                            easyadapt=easyadapt,
                            tgtcorpus=tgtcorpus)
    try:
        CMP.make_fvectors()
    except Exception, e:
        print pformat(e)
        traceback.print_exc(file=sys.stdout)
        raise
    finally:
        del(CMP)


#def make_fvectors(verbcorpus_dir=None, verbset_path=None, dataset_dir=None, restart_from=None, f_types=None):
#    CM = CaseMaker(verbcorpus_dir=verbcorpus_dir, verbset_path=verbset_path, dataset_dir=dataset_dir, restart_from=restart_from, f_types=f_types)
#    CM.make_fvectors()



#----------------------------------------------------------------------------------------------------
# Classifier wrappers for bolt and scikit-learn linear model
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
            self.multicpu = self.opts["multicpu"]
        if "shuffle" in self.opts:
            self.shuffle = True
        if "kernel_approximation" in self.opts:
            self.kernel_approx = True
        else:
            self.kernel_approx = False

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
        self.dspath = dataset_path if dataset_path is not None else "invalid path!"
        try:
            self.X = load_sparse_matrix(os.path.join(dataset_path, "X.npz"))
            self.Y = np.load(os.path.join(dataset_path, "Y.npy"))
        except IOError:
            print "Seems model files (X.npz, Y.npy) are not found..."

    def trainSGD(self):
        sgd = SGDClassifier(loss=self.loss, penalty=self.reg, alpha=self.alpha, n_iter=self.epochs,
                            shuffle=True, n_jobs=self.multicpu, class_weight='auto')
        # print "Classifier (sklearn SGD): training the model \t(%s)"%self.dspath
        if self.kernel_approx is True:
            rbf_feature = RBFSampler(gamma=1, n_components=100.0, random_state=1)
            Xk = rbf_feature.fit_transform(self.X)
            self.glm = OneVsRestClassifier(sgd).fit(Xk, self.Y)
        else:
            self.glm = OneVsRestClassifier(sgd).fit(self.X, self.Y)
        print "Classifier (sklearn SGD): Done. \t(%s)"%self.dspath

    def trainSVM(self):
        # svm = NuSVC(nu=0.5, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=500, verbose=False, max_iter=-1)
        svm = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1)
        print "Classifier (sklearn NuSVC): training the model \t(%s)"%self.dspath
        self.glm = OneVsRestClassifier(svm).fit(self.X, self.Y)
        print "Classifier (sklearn NuSVC): Done. \t(%s)"%self.dspath


    def predict_f(self, testset_path=None, X=None, Y=None):
        fn_x = os.path.join(testset_path, "X.npz")
        fn_y = os.path.join(testset_path, "Y.npy")
        Xtest = load_sparse_matrix(fn_x)
        Ytest = np.load(fn_y)
        pred = self.glm.predict(Xtest)
        return pred

    def predict_f_prob(self, testset_path=None, X=None, Y=None):
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



def _selftest_sk(modelpath="", dspath=""):
    X = load_sparse_matrix(os.path.join(dspath, "X.npz"))
    glm = pickle.load(open(modelpath, "rb"))
    Y = np.load(os.path.join(dspath, "Y.npy"))
    pred = glm.predict(X)
    from sklearn.metrics import classification_report
    print classification_report(Y, pred)



def train_sklearn_classifier_batch(dataset_dir="", modeltype="sgd_maxent_l2", verbset_path="", selftest=False, 
                                   cls_option={"loss":"log", "epochs":10, "alpha":0.0001, "reg":"L2"},
                                   pool_num=2):
    vs_file = pickle.load(open(verbset_path, "rb"))
    verbs = vs_file.keys()
    verbsets = deepcopy(vs_file)
    set_names = [os.path.join(dataset_dir, v) for v in verbs]
    po = Pool(processes=pool_num, maxtasksperchild=16)
    args = []
    for idd, dir in enumerate(set_names):
        dspath = os.path.join(dir)
        arg = {"dataset_dir":dspath, "output_path":dir, "modeltype":modeltype, "options":cls_option}
        args.append(arg)
    po.map(train_sklearn_classifier_p, args)
    po.close()
    po.join()


def train_sklearn_classifier_p(args={}):
    modeltype = args["modeltype"]
    cls_option = args["options"]
    dataset_dir = args["dataset_dir"]
    output_path = args["output_path"]
    classifier = SklearnClassifier(mtype=modeltype, opts=cls_option)
    classifier.setopts()
    classifier.load_dataset(dataset_dir)
    if "sgd" in modeltype:
        classifier.trainSGD()
    elif "svm" in modeltype:
        classifier.trainSVM()
    modelfilename = os.path.join(output_path, "model_%s.pkl2"%modeltype)
    classifier.save_model(modelfilename)


# def train_sklearn_classifier(dataset_dir="", output_path="", modeltype="sgd", 
#                             cls_option={"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}):
#     classifier = SklearnClassifier(mtype=modeltype, opts=cls_option)
#     classifier.setopts()
#     classifier.load_dataset(dataset_dir)
#     if modeltype == "sgd":
#         classifier.trainSGD()
#     modelfilename = os.path.join(output_path, "model_%s.pkl2"%modeltype)
#     classifier.save_model(modelfilename)
#     # _selftest_sk(modelfilename, dataset_dir)

# def train_sklearn_classifier_batch(dataset_dir="", modeltype="sgd", verbset_path="", selftest=False, 
#                                    cls_option={"loss":"hinge", "epochs":10, "alpha":0.0001, "reg":"L2"}):
#     vs_file = pickle.load(open(verbset_path, "rb"))
#     verbs = vs_file.keys()
#     verbsets = deepcopy(vs_file)
#     set_names = [os.path.join(dataset_dir, v) for v in verbs]
#     for idd, dir in enumerate(set_names):
#         dspath = os.path.join(dir)
#         print "Batch trainer (sklearn %s):started\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
#         train_sklearn_classifier(dataset_dir=dspath, output_path=dir, modeltype=modeltype, cls_option=cls_option)
#         print "Batch trainer (sklearn %s):done!\t dir= %s (%d out of %d)"%(modeltype, dir, idd+1, len(set_names))
#         if selftest:
#             print "Batch trainer selftest..."
#             _selftest_sk(modelfilename, dspath)
#             print "Batch trainer selftest... done!"



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


def train_bolt_classifier_batch(dataset_dir="", output_dir="", modeltype="sgd_maxent_l2", verbset_path="", selftest=False, 
                                   cls_option={"loss":"log", "epochs":10, "alpha":0.0001, "reg":"L2"},
                                   pool_num=2):
    vs_file = pickle.load(open(verbset_path, "rb"))
    verbs = vs_file.keys()
    verbsets = deepcopy(vs_file)
    set_names = [os.path.join(dataset_dir, v) for v in verbs]
    po = Pool(processes=pool_num, maxtasksperchild=16)
    args = []
    for idd, dir in enumerate(set_names):
        dspath = os.path.join(dir)
        outputpath = os.path.join(output_dir, dir)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        arg = {"dataset_dir":dspath, "output_path":outputpath, "modeltype":modeltype, "options":cls_option}
        args.append(arg)
    po.map(train_bolt_classifier_p, args)
    po.close()
    po.join()


def train_bolt_classifier_p(args={}):
    modeltype = args["modeltype"]
    cls_option = args["options"]
    dataset_dir = args["dataset_dir"]
    output_path = args["output_path"]
    classifier = BoltClassifier(mtype=modeltype, opts=cls_option)
    classifier.load_dataset(dataset_dir)
    if "sgd" in modeltype:
        classifier.trainSGD()
    else:
        raise NotImplementedError
    modelfilename = os.path.join(output_path, "model_%s.pkl2"%modeltype)
    classifier.save_model(modelfilename)



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
