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
from copy import deepcopy
from multiprocessing import Pool
from tool.betterpool import Pool2
from pattern.text import en
from nose.plugins.attrib import attr
import random
tense = ["1sg", "3sg", "pl", "past"]



def is_verbincluded_p(sent_conjlist_pair=None):
    """
    This is for checking whether given sentence contains any of conjugation of the verb
    Parameters
    -----------
        corpus_conjlist_pair: ("")
    @returns
        vflag (True if the verb is in sentence else False)
    """
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

def is_verbincluded2(sent="", conjs=[]):
    """
    This is for checking whether given sentence contains any of conjugation of the verb
    @args
        sentence(as string), conjs(conjugation of the verb)
    @returns
        vflag (True if the verb is in sentence else False)
    """
    vflag = False
    if sent:
        try:
            for c in conjs:
                if c in sent:
                    vflag = True
                    break
        except IndexError, e:
            pass
    return vflag

def _get_conjs(verb=""):
    """
    Generate possible conjugation of given verb
    """
    return [en.conjugate(verb, c)+"\t_\t_\tVB" for c in tense]


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
    return list(allv), vconj_dic


def _extract_sents(corpus=[], verb="", sample_max_num = 10000, conjs = []):
    """
    this will extract training sentences (parsed file format) into given output directory
    """
    v_corpus = []
    try:
        for sid, sentence in enumerate(corpus):
            if is_verbincluded2(sentence, conjs):
                s = sentence.split("\n") 
                v_corpus.append(s)
            elif sid % 10000 == 0:
                if len(v_corpus) >= sample_max_num:
                    break
            else:
                pass
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    return v_corpus


def _save_vcorpus(verb="", v_corpus=[], output_dir=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, verb+".pkl2")
    with open(filename, "wb") as pf:
        pickle.dump(v_corpus, pf)
    print "IO: Pickling %d sentences containing verb '%s'"%(len(v_corpus), verb)
    print "IO: File %s is saved.\n\n"%filename


def _save_vcorpusdic(vcorpusdic={}, output_dir=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for vn, vc in vcorpusdic.iteritems():
        filename = os.path.join(output_dir, vn+".pkl2")
        with open(filename, "wb") as pf:
            print "IO: Pickling %d sentences containing verb '%s'"%(len(vc), vn)
            pickle.dump(vc, pf, -1)
        print "IO: File %s is saved.\n\n"%filename


def extract_sentence_for_verbs(ukwac_prefix = "", output_dir="",
                               verbset_path = "", sample_max_num = 10000, shuffle=True):
    """
    A wrapper for extracting parsed sentences which containing the verbs in given verbset
    """
    import glob
    ukwacfiles = glob.glob(ukwac_prefix+"*.parsed")
    raw_verbset = pickle.load(open(verbset_path, "rb"))
    verbset = deepcopy(raw_verbset)
    verbs, conjdic = _retrieve_unique_verbs(verbset)
    output_dic = defaultdict(list)
    if shuffle is True:
        random.seed(output_dir+verbset_path)
        random.shuffle(ukwacfiles)
    try:
        for fc, file in enumerate(ukwacfiles):
            print "IO: Reading corpus.... file count %d"%fc
            with open(file, "r") as cf:
                corpus = cf.read().split("\n\n")
                print "IO: Reading corpus... done!"
            try:
                for v in verbs:
                    conjlist = conjdic[v]
                    if len(output_dic[v]) > sample_max_num:
                        output_dic[v] = output_dic[v][:sample_max_num]
                        verbs.remove(v)
                    if fc >= 5:
                        raise CorpusFileCountOverlimit
                    print "Extraction: verb = '%s' (%d remaining)"%(v, len(verbs)), "\t\tworking on file %s"%file
                    output_dic[v] += _extract_sents(corpus, v, sample_max_num, conjlist)
            except CorpusFileCountOverlimit:
                raise CorpusFileCountOverlimit
    except KeyboardInterrupt:
        print "Interrupted by user... aborting"
    except CorpusFileCountOverlimit:
        print "Reached file count limitation... aborting"
    finally:
        _save_vcorpusdic(output_dic, output_dir)
    print "Extracting sentences: done"

class CorpusFileCountOverlimit(Exception):
    pass


def extract_parallel(ukwac_prefix = "", output_dir="",
                     verbset_path = "", sample_max_num = 10000, shuffle=True):
    """
    A wrapper for extracting parsed sentences which containing the verbs in given verbset
    """
    import glob
    ukwacfiles = glob.glob(ukwac_prefix+"*.parsed")
    raw_verbset = pickle.load(open(verbset_path, "rb"))
    verbset = deepcopy(raw_verbset)
    verbs, conjdic = _retrieve_unique_verbs(verbset)
    output_dic = defaultdict(list)
    if shuffle is True:
        random.seed(output_dir+verbset_path)
        random.shuffle(ukwacfiles)
    try:
        for fc, file in enumerate(ukwacfiles):
            print "IO: Reading corpus.... file count %d"%fc
            with open(file, "r") as cf:
                corpus = cf.read().split("\n\n")
                print "IO: Reading corpus... done!"
            try:
                p = Pool(4)
                args = [(corpus, conjdic[v], sample_max_num) for v in verbs]
                result = p.map(_extract_sents_p, args)
                output_dic = {v:item for v, item in zip(verbs, result)}
                # for v in verbs:
                #     conjlist = conjdic[v]
                #     if len(output_dic[v]) > sample_max_num:
                #         output_dic[v] = output_dic[v][:sample_max_num]
                #         verbs.remove(v)
                if fc >= 5:
                    raise CorpusFileCountOverlimit
                    # print "Extraction: verb = '%s' (%d remaining)"%(v, len(verbs)), "\t\tworking on file %s"%file
                # output_dic[v] += _extract_sents(corpus, v, sample_max_num, conjlist)
            except CorpusFileCountOverlimit:
                raise CorpusFileCountOverlimit
    except KeyboardInterrupt:
        print "Interrupted by user... aborting"
    except CorpusFileCountOverlimit:
        print "Reached file count limitation... aborting"
    finally:
        _save_vcorpusdic(output_dic, output_dir)
    print "Extracting sentences: done"

def _extract_sents_p(args=None):
    """
    this will extract training sentences (parsed file format) into given output directory
    """
    corpus = args[0]
    conjs = args[1]
    sample_max_num = args[2]
    v_corpus = []
    try:
        for sid, sentence in enumerate(corpus):
            if is_verbincluded2(sentence, conjs):
                s = sentence.split("\n") 
                v_corpus.append(s)
            elif sid % 10000 == 0:
                if len(v_corpus) >= sample_max_num:
                    break
            else:
                pass
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    return v_corpus


@attr("extract_tiny")
def test_extract_small():
    extract_sentence_for_verbs(ukwac_prefix="../sandbox/classify/", output_dir="../sandbox/classify/test_out",
                                verbset_path="../sandbox/classify/vs_tiny.pkl2", sample_max_num=1000)

@attr("extract_p")
def test_extract_parallel():
    extract_parallel(ukwac_prefix="../sandbox/classify/", output_dir="../sandbox/classify/test_out",
                                verbset_path="../sandbox/classify/vs_tiny.pkl2", sample_max_num=1000)



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
