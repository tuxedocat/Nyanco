#!/usr/bin/env python
#coding: utf-8
'''
Nyanco/src/detector.py
Created on 1 Jan 2013
'''
__author__ = "Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os
import glob
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from copy import deepcopy
from multiprocessing import Pool
from pattern.text import en
from nose.plugins.attrib import attr
import random
import errno
tense = ["1sg", "3sg", "pl", "past"]


def _get_conjs(verb=""):
    """
    Generate possible conjugation of given verb
    """
    return [" "+en.conjugate(verb, c)+" " for c in tense]


def is_verbinline(l, conjs):
    flag = False
    for c in conjs:
        if c in l:
            flag = True
            break
    return flag


def save_vcorpus(v, rawsents, output_dir):
    with open(os.path.join(output_dir, v+".txt"), "w") as f:
        f.writelines(rawsents)

class CorpusFileCountOverlimit(Exception):
    pass

class LengthOverlimit(Exception):
    pass

def extract_rawsents(args):
    """
    A wrapper for extracting parsed sentences which containing the verbs in given verbset
    """
    ukwac_prefix = args["ukwac_prefix"]
    v = args["v"]
    output_dir = args["output_dir"]
    sample_max_num = args["sample_max_num"]
    ukwacfiles = glob.glob(ukwac_prefix+"*")
    conjs = _get_conjs(v)
    random.shuffle(ukwacfiles)
    rawsents = []
    try:
        for fc, file in enumerate(ukwacfiles):
            with open(file, "r") as cf:
                corpus = cf.readlines()
            for l in corpus:
                if is_verbinline(l, conjs):
                    rawsents.append(l)
                    if len(rawsents) > sample_max_num:
                        raise LengthOverlimit
            # if fc > 500:
            #     raise CorpusFileCountOverlimit
    except KeyboardInterrupt:
        print "Interrupted by user... aborting"
    except CorpusFileCountOverlimit:
        print "Reached file count limitation... aborting"
    except LengthOverlimit:
        print "Reached line num. limitation of extracted file... aborting"
    finally:
        save_vcorpus(v, rawsents, output_dir)
    print "Extracting sentences: done"


def extract_raw(ukwac_prefix=None, verbset_path=None, output_dir=None, pool_num=24, sample_max_num=30000):
    args = []
    try:
        os.makedirs(os.path.abspath(output_dir))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(verbset_path, "rb") as f:
        vs_full = pickle.load(f)
    sep_keys = [w for w in vs_full]
    random.shuffle(sep_keys)
    for v in sep_keys:
        args.append({"ukwac_prefix":ukwac_prefix, 
                     "output_dir":output_dir, 
                     "v":v, 
                     "sample_max_num":sample_max_num})
    print args
    mp = Pool(processes=pool_num, maxtasksperchild=1)
    mp.map(extract_rawsents, args)
    mp.close()
    mp.join()



if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description =   """
                    python examples_extractor_senna.py -c ../sandbox/classify/tiny/ -o ../sandbox/classify/tiny/out -v ./tool/verbset_111_20.pkl2 -n 3000
                    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--ukwac", action="store", 
                    help="path to ukWaC corpus (raw)")
    ap.add_argument("-o", '--output_dir', action="store",
                    help="path to output directory")
    ap.add_argument("-v", '--verbset', action="store",
                    help="path of verbset pickle file")
    ap.add_argument("-n", '--maximum_num', action="store", type=int,
                    help="max number of sentence that you need to collect")
    args = ap.parse_args()

    if args.ukwac:
        extract_raw(ukwac_prefix=args.ukwac, 
                    verbset_path=args.verbset, 
                    output_dir=args.output_dir, 
                    sample_max_num=args.maximum_num)
    else:
        ap.print_help()
    quit()
