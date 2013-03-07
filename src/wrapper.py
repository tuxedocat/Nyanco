#!/usr/bin/env python
#coding:utf-8
'''
Nyanco/src/wrapper.py
Created on 15 Nov. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import os, glob
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
import yaml
from copy import deepcopy
import cProfile as profile
import pstats
import examples_extractor
import classifier
import detector


class Experiment(object):
    def __init__(self, name="", conf={}):
        print pformat(conf)
        if not "name":
            raise Exception
        self.name = name
        conf.update({"confname": name})
        self.c = conf
        self.pl = conf["pipeline"]
        self.vs = conf["confusion_set"]
        self.toolkit = conf["toolkit"]
        self.vsname = os.path.basename(self.vs).split(".")[0]
        self.cls_opts = conf["classifier_args"]
        self.parallel_num = conf["parallel_num"] if "parallel_num" in conf else 4 
        print "Num. of Parallel Processes is ", self.parallel_num
        if "extract_examples" in self.pl:
            self.native_c = conf["dir_ukwac"]
            self.numts = conf["num_tsamples"]
            self.ext_dir = os.path.join(conf["dir_out"], self.vsname+"_%s"%(str(self.numts)))
        if "make_features" in self.pl:
            self.features = conf["features"]
            try:
                self.vcdir = self.ext_dir
            except:
                self.vcdir = conf["verbcorpus_dir"]
            self.dsdir = conf["dir_models"] 
            self.numts = conf["num_tsamples"]
            self.easyadapt = True if "easyadapt" in conf else False
            self.tgtcorpus_path = conf["tgtcorpus"] if "tgtcorpus" in conf else None
        if "train" in self.pl:
            self.dsdir = conf["dir_models"] 
            self.modeldir = conf["dir_models"]
            self.model = conf["classifier"]
            self.cls_opts = conf["classifier_args"]
        if "detect" in self.pl:
            self.cls_opts = conf["classifier_args"]
            self.model = conf["classifier"]
            self.dsdir = conf["dir_models"]
            self.features = conf["features"]
            self.dir_log = conf["dir_log"]
            self.dtype = conf["detector"]
            self.dopt = conf["detector_options"]
            self.fcepath = conf["fce_path"]


    def execute(self):
        # pstats_fn = self.dir_log + datetime.now().strftime("./pstats_%Y%m%d_%H%M.stats")
        if "extract_examples" in self.pl:
            examples_extractor.extract_sentence_for_verbs(ukwac_prefix=self.native_c,
                                                         output_dir=self.ext_dir,
                                                         verbset_path=self.vs,
                                                         sample_max_num=self.numts,
                                                         shuffle=True)
        if "make_features" in self.pl:
            classifier.make_fvectors(verbcorpus_dir=self.vcdir,
                                     verbset_path=self.vs,
                                     dataset_dir=self.dsdir,
                                     f_types=self.features,
                                     pool_num=self.parallel_num, 
                                     instance_num=self.numts,
                                     easyadapt=self.easyadapt, 
                                     tgtcorpus_path=self.tgtcorpus_path)
        if "train" in self.pl:
            if self.toolkit == "bolt":
                classifier.train_boltclassifier_batch(dataset_dir=self.dsdir, 
                                                 modeltype=self.model, 
                                                 verbset_path=self.vs, 
                                                 selftest=False, 
                                                 cls_option=self.cls_opts)
            elif self.toolkit == "sklearn":
                classifier.train_sklearn_classifier_batch(dataset_dir=self.dsdir, 
                                                modeltype=self.model, 
                                                verbset_path=self.vs, 
                                                selftest=False, 
                                                cls_option=self.cls_opts,
                                                pool_num=self.parallel_num)
        if "detect" in self.pl:
            if "classifier" in self.dtype:
                k = self.dopt["ranker_k"] if "ranker_k" in self.dopt else 5
                gs_k = self.dopt["grid_k"] if "grid_k" in self.dopt else []  # grid search for ranker_k ex.[1, 5, 10, 20]
                if "kbest" in self.dtype:
                    d_algo = "kbest"
                elif "rank" in self.dtype:
                    d_algo = "ranker"
                elif "conf" in self.dtype:
                    d_algo = "confidence"
                else:
                    d_algo = "suddendeath"
                log_conf = {"__exp_name__": self.name, "features": str(self.features), 
                            "model": str(self.model) + "::" + str(self.cls_opts),
                            "datapath_fce": self.fcepath, "datapath_models": self.dsdir,
                            "confusionset": self.vs, "detector_info": self.dtype + " (k=%d)"%k}
                if gs_k:
                    detector.detectmain_c_gs(corpuspath=self.fcepath, 
                                          model_root=self.dsdir, 
                                          type=self.model, 
                                          reportout=self.dir_log, 
                                          verbsetpath=self.vs,
                                          d_algo=d_algo,
                                          ls_ranker_k=gs_k,
                                          features=self.features,
                                          expconf=log_conf)
                else:
                    detector.detectmain_c(corpuspath=self.fcepath, 
                                          model_root=self.dsdir,
                                          type=self.model, 
                                          reportout=self.dir_log, 
                                          verbsetpath=self.vs,
                                          d_algo=d_algo,
                                          ranker_k=k,
                                          features=self.features,
                                          expconf=log_conf)
            elif self.dtype == "lm":
                lmpath = self.dopt["LM_path"] if "LM_path" in self.dopt else ""
                paslmpath = self.dopt["PASLM_path"] if "PASLM_path" in self.dopt else ""
                detect.detectmain(corpuspath=self.fcepath, 
                                  reportout=self.dir_log, 
                                  lmpath=self.lmpath, 
                                  paslmpath=args.pas_lm_path, 
                                  verbsetpath=args.verbset)


class Config(object):
    """
    This class reads+holds configurations of experiments
    """
    def __init__(self, path):
        """
        path: path-to yaml file
        """
        f = open(path, 'r')
        self.conf = yaml.safe_load(f)


def do_experiments(path_to_yaml=""):
    if path_to_yaml:
        conf = Config(path_to_yaml).conf
        names = sorted(conf.keys())
        for ex in names:
            cf = conf[ex]
            exp = Experiment(name=ex, conf=cf)
            exp.execute()
    else:
        print "please give me a valid path of config file!"


if __name__ == "__main__":
    import sys
    do_experiments(sys.argv[1])
