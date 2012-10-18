# ! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/detector.py
Created on 18 Oct 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from numpy import array
from nltk.corpus import wordnet as wn
from pattern.text import en
from nltk import WordNetLemmatizer
from nose.plugins.attrib import attr
import random
try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *
import tool.altword_generator as altgen
from sklearn import cross_validation
tense = ["1sg", "3sg", "pl", "past"]


class ClassifierExample(object):
    pass


def is_verbincluded(verb="", sent=[]):
    i_suf = 1
    i_pos = 4
    i_ne = 10
    conjs = ["1sg", "3sg", "pl", "past"]
    v_conjs = [en.conjugate(verb, c) for c in conjs]
    tags = [tuple(l.split("\t")) for l in sent]
    vflag = False
    if tags:
        for tt in tags:
            try:
                if tt[i_suf] in v_conjs and "VB" in tt[i_pos]:
                    vflag = True
            except IndexError, e:
                pass
    return vflag

def is_verbincluded2(verb="", sent="", conjs=[]):
    vflag = False
    cand = [c+"\t_\t_\tVB" for c in conjs]
    if sent:
        try:
            for c in cand:
                if c in sent:
                    vflag = True
                    break
        except IndexError, e:
            pass
    return vflag

def _get_conjs(verb=""):
    return [en.conjugate(verb, c) for c in tense]


def _is_dic_len_over(dic={}, max_len=0):
    flag = False
    for k in dic:
        if len(dic[k]) > max_len:
            flag = True
    return flag


def _retrieve_unique_verbs(verbset={}):
    all_v = []
    vconj_dic = defaultdict(list)
    for k, v in verbset.iteritems():
        all_v.append(k)
        all_v += [vt[0] for vt in v]
    allv = set(all_v)
    for v in allv:
        vconj_dic[v] = _get_conjs(v)
    return allv, vconj_dic

def _extract_sents(corpus=[], verb="", sample_max_num = 10000, conjs = []):
    """
    this will extract training sentences (parsed file format) into given output directory
    """
    v_corpus = []
    try:
        for sentence in corpus:
            s = sentence.split("\n") 
            # if is_verbincluded(verb, s) and len(v_corpus) < sample_max_num:
            #     v_corpus.append(s)
            if is_verbincluded2(verb, sentence, conjs) and len(v_corpus) < sample_max_num:
                v_corpus.append(s)
            elif len(v_corpus) >= sample_max_num:
                break
            else:
                pass
    except KeyboardInterrupt:
        print "Interrupted by user... aborting"
    return v_corpus

def _save_vcorpus(verb="", v_corpus=[], output_dir=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, verb+".pkl2")
    with open(filename, "wb") as pf:
        pickle.dump(v_corpus, pf)
    print "IO: Pickling %d sentences containing verb '%s'"%(len(v_corpus), verb)
    print "IO: File %s is saved.\n\n"%filename


def extract_sentence_for_verbs(ukwac_prefix = "", output_dir="",
                               verbset_path = "", sample_max_num = 10000, shuffle=True):
    """
    A wrapper for extracting parsed sentences which containing the verbs in given verbset
    """
    import glob
    ukwacfiles = glob.glob(ukwac_prefix+"*.parsed")
    raw_verbset = pickle.load(open(verbset_path, "rb"))
    verbset = raw_verbset["verbset"]
    verbs, conjdic = _retrieve_unique_verbs(verbset)
    if shuffle is True:
        random.seed(output_dir)
        random.shuffle(ukwacfiles)
    for vid, v in enumerate(verbs):
        try:
            v_corpus = []
            conjlist = conjdic[v]
            for file in ukwacfiles:
                with open(file, "r") as cf:
                    corpus = cf.read().split("\n\n")
                print "verb: '%s' (%d out of %d)"%(v, vid+1, len(verbs)), "\t\tworking on file %s"%file
                v_corpus += _extract_sents(corpus, v, sample_max_num, conjlist)
                if len(v_corpus) > sample_max_num:
                    break
        except CorpusLengthOverlimit:
            pass
        finally:
            if v_corpus:
                _save_vcorpus(v, v_corpus, output_dir)
    print "Extracting sentences: done"

class CorpusLengthOverlimit(Exception):
    pass


def extract_training_examples(ukwac_prefix = "", verbset_path = "", max_num = 10000, shuffle=False, top_n_verbs=30):
    """
    Extract training examples from given (parsed and separated) ukWaC corpus and given verbset

    Extracted instances will be saved under given output path, like following:
        set of verb1: outputprefix/verb1/dataset_verb1.pkl2

    @args
        ukwac_prefix: is prefix of the path, to parsed ukwac files
        verbset_path: is prefix of the path, to pickled verbset
        max_num: is a number, that restrict the maximum numbers of each verb's sample
        shuffle: if True, corpus files will be shuffled
    @returns
        None
    """
    import glob
    filelist = glob.glob(ukwac_prefix+"*.parsed")
    verbset = pickle.load(open(verbset_path, "rb"))
    verblist = verbset["verbs"]
    vset = verbset["verbset"]
    for verb in verblist:
        for cfname in filelist:
            corpus = open(cfname, 'r').read().split("\n\n")
            corpus = [s.split("\n") for s in corpus]


    return None


@attr("extract_tiny")
def test_extract_small():
    extract_sentence_for_verbs(ukwac_prefix="../sandbox/classify/tiny/", output_dir="../sandbox/classify/tiny/out",
                                verbset_path="./tool/verbset_111_20.pkl2", sample_max_num=3)


if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description =   """
                    python examples_extractor.py -c ../sandbox/classify/tiny/ -o ../sandbox/classify/tiny/out -v ./tool/verbset_111_20.pkl2 -n 3000
                    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--ukwacparsed_path", action="store", 
                    help="path to ukWaC corpus (parsed and separated)")
    ap.add_argument("-o", '--output_dir', action="store",
                    help="path to output directory")
    ap.add_argument("-v", '--verbset', action="store",
                    help="path of verbset pickle file")
    ap.add_argument("-n", '--maximum_num', action="store", type=int,
                    help="max number of sentence that you need to collect")
    ap.add_argument("-s", '--shuffle', action="store_true",
                    help="if you want to shuffle the corpus files...")
    args = ap.parse_args()

    if (args.ukwacparsed_path):
        extract_sentence_for_verbs(ukwac_prefix=args.ukwacparsed_path, output_dir=args.output_dir,
                                    verbset_path=args.verbset, sample_max_num=args.maximum_num, shuffle=args.shuffle)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    else:
        ap.print_help()
    quit()