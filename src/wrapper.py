# ! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/wrapper.py
Created on 15 Nov. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
# import logging
# logfilename = datetime.now().strftime("config_log_%Y%m%d_%H%M.log")
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../../log/'+logfilename)
import os, glob
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
import yaml
from copy import deepcopy

import examples_extractor
import classifier
import detector


class Experiment(object):
    def __init__(self, name="", conf={}):
        print pformat(conf)
        if not "name":
            raise Exception
        self.name = name
        self.c = conf
        self.pl = conf["pipeline"]
        self.vs = conf["verbset"]
        self.toolkit = conf["toolkit"]
        self.vsname = os.path.basename(self.vs).split(".")[0]
        self.cls_opts = conf["classifier_args"]
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
            self.dsdir = os.path.join(conf["dir_train"], self.name) 
        if "train" in self.pl:
            self.dsdir = os.path.join(conf["dir_train"], self.name) 
            self.model = conf["classifier"]
            self.cls_opts = conf["classifier_args"]
        if "detect" in self.pl:
            self.model = conf["classifier"]
            self.dsdir = os.path.join(conf["dir_train"], self.name) 
            self.dir_log = os.path.join(conf["dir_log"], self.name)
            self.dtype = conf["detector"]
            self.dopt = conf["detector_options"]
            self.fcepath = conf["fce_path"]


    def execute(self):
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
                                     f_types=self.features)
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
                                                 cls_option=self.cls_opts)
        if "detect" in self.pl:
            if "classifier" in self.dtype:
                k = self.dopt["ranker_k"] if "ranker_k" in self.dopt else 5
                d_algo = "kbest" if "kbest" in self.dtype else "suddendeath"
                detector.detectmain_c(corpuspath=self.fcepath, 
                                      model_root=self.dsdir, 
                                      type=self.model, 
                                      reportout=self.dir_log, 
                                      verbsetpath=self.vs,
                                      d_algo=d_algo,
                                      ranker_k=k)
            elif self.dtype == "lm":
                lmpath = self.dopt["LM_path"] if "LM_path" in self.dopt else ""
                paslmpath = self.dopt["PASLM_path"] if "PASLM_path" in self.dopt else ""
                detect.detectmain(corpuspath=self.fcepath, 
                                  reportout=self.dir_log, 
                                  lmpath=self.lmpath, 
                                  paslmpath=args.pas_lm_path, 
                                  verbsetpath=args.verbset)

        print "done!"


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
        names = conf.keys()
        for ex in names:
            cf = conf[ex]
            exp = Experiment(name=ex, conf=cf)
            exp.execute()
    else:
        print "please give me a valid path of config file!"


if __name__ == "__main__":
    import sys
    do_experiments(sys.argv[1])
