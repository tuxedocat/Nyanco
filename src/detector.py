#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/detector.py
Created on 9 Sep. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

import os
import nltk
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from datetime import datetime
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)
from nltk import ngrams
try:
    from lsa_test.irstlm import *
except:
    pass


class DetectorBase(object):
    def __init__(self, corpusdictpath):
        if os.path.exists(corpusdictpath):
            with open(corpusdictpath, "rb") as f:
                corpusdict = pickle.load(f)
                self.corpus = corpusdict
            self.experimentset = defaultdict(dict)
        else:
            raise IOError

    def make_cases(self):
        """
        Makes test instances entirely on the corpus, just a wrapper
        """
        self.testcases = defaultdict(dict)
        self.case_keys = []
        for docname, doc in self.corpus.iteritems():
            try:
                self._mk_cases(docname, doc)
            except KeyError as ke:
                logging.debug(pformat(ke))
            
            except Exception as e:
                logging.debug("error catched in make_cases")
                logging.debug(pformat(e))

    def detect(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError



class LM_Detector(DetectorBase):
    def read_LM_and_PASLM(self, path_IRSTLM="", path_PASLM=""):
        if path_IRSTLM:
            self.LM = initLM(5, path_IRSTLM)
            logging.debug(pformat("IRSTLM's LM is loaded from %s"%path_IRSTLM))
        if path_PASLM:
            self.pasCounter = pickle.load(open(path_PASLM))
            logging.debug(pformat("PASLM is loaded"))
            self.paslm_c_sum = sum(self.pasCounter.values())



    def _mk_ngram_queries(self, n=5, cp_pos=None, w_list=[], alt_candidates=[]):
        """
        Make a query for ngram frequency counter
            nltk.ngrams returns when `pad_right=True, pad_symbol=" "`
                [('I', 'have', 'a', 'black', 'cat.'),
                ('have', 'a', 'black', 'cat.', ' '), 
                ('a', 'black', 'cat.', ' ', ' '), 
                ('black', 'cat.', ' ', ' ', ' '), 
                ('cat.', ' ', ' ', ' ', ' ')]
        First, find index of checkpoint `a` (2), then for n=5, index of the desired tuple is `cp_pos - (n-1)/2`
        @takes:
            n :: N gram size (if n=2, [-2 -1 word +1 +2])
            cp_pos:: int, positional index of the checkpoint
            w_list:: list, words of a sentence
            alt_candidates:: list, alternative candidates if given
        @returns:
            org_q:: string, queries for irstlm.getSentenceScore (Original word)
            alt_q:: list of string, queries for irstlm.getSentenceScore (Generated by given candidates)
            moc_smart_alt_q:: list of string, ad hoc moc of Smartquery
                                    just concatenate heads 
        """
        org_q = []
        alt_q = []
        try:
            ngrams_l = ngrams(w_list, n=n, pad_right=True, pad_symbol=" ")
            query = ngrams_l[cp_pos - (n-1)/2]
            if alt_candidates:
                for cand in alt_candidates:
                    tmp = list(query[:])
                    tmpi = int((n - 1)/2)
                    tmp.pop(tmpi)
                    tmp.insert(tmpi, cand)
                    alt_q.append(str(" ".join(tmp)))
            else:
                alt_q.append(str(" ".join(query)))
            org_q.append(str(" ".join(query)))
        except Exception as nge:
            logging.debug(format(w_list, cp_pos))
            logging.debug(pformat(nge))
        return org_q, alt_q


    def _mk_PAS_queries(self, pasdiclist=[], org_preds=[], alt_preds=[]):
        org_pas_q = []
        alt_pas_q = []
        for pdic in pasdiclist:
            PRED = pdic["PRED"][0]
            ARG0 = pdic["ARG0"][0]
            ARG1 = pdic["ARG1"][0]
            tmp = (PRED, ARG0, ARG1)
            if PRED in org_preds:
                org_pas_q.append(tmp)
            if PRED in alt_preds:
                alt_pas_q.append(tmp)
        org_pas_q = list(set(org_pas_q))
        alt_pas_q = list(set(alt_pas_q))
        if org_pas_q and alt_pas_q:
            logging.debug(pformat(org_pas_q))
            logging.debug(pformat(alt_pas_q))
        return org_pas_q, alt_pas_q


    def _mk_cases(self, docname="", doc=None):
        if docname and doc:
            try:
                gold_tags = doc["gold_tags"]
                test_tags = doc["RVtest_tags"]
                gold_text = doc["gold_text"]
                test_text = doc["RVtest_text"]
                gold_words = doc["gold_words"]
                test_words = doc["RVtest_words"]
                gold_pas = doc["gold_PAS"]
                test_pas = doc["RVtest_PAS"]
                checkpoints = doc["errorposition"]
                for cpid, cp in enumerate(checkpoints):
                    testkey = docname+"_checkpoint"+str(cpid)
                    self.case_keys.append(testkey)
                    cp_pos = cp[0]
                    incorr = cp[1]
                    gold = cp[2]
                    test_wl = test_words[cpid]
                    query_altwords = [gold]
                    self.testcases[testkey]["checkpoint_idx"] = cp_pos
                    self.testcases[testkey]["incorrect_label"] = incorr
                    self.testcases[testkey]["gold_label"] = gold
                    org_qs, alt_qs = self._mk_ngram_queries(n=5, cp_pos=cp_pos, w_list=test_wl, alt_candidates=query_altwords)
                    self.testcases[testkey]["LM_queries"] = {"org":org_qs, "alt":alt_qs}
                    org_pqs, alt_pqs = self._mk_PAS_queries(pasdiclist=gold_pas+test_pas, org_preds=[incorr], alt_preds=query_altwords)
                    self.testcases[testkey]["PASLM_queries"] = {"org":org_pqs, "alt":alt_pqs}

            except Exception as e:
                logging.debug("error catched in _mk_cases")
                logging.debug(pformat(testkey))
                logging.debug(pformat(e))

    def LM_count(self):
        """
        calculate scores of given string query, using irstlm.getSentenceScore
        """
        for testid in self.case_keys:
            case = self.testcases[testid]
            self.testcases[testid]["LM_scores"] = {"org":[], "alt":[]}
            for org_q in case["LM_queries"]["org"]:
                logging.debug(pformat(org_q))
                try:
                    score = getSentenceScore(self.LM, org_q)
                    logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["LM_scores"]["org"].append(score)
            for alt_q in case["LM_queries"]["alt"]:
                logging.debug(pformat(alt_q))
                try:
                    score = getSentenceScore(self.LM, alt_q)
                    logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["LM_scores"]["alt"].append(score)


    def _getPASLMscore(self, pasCounter={}, pas_q=[]):
        """
        Get count and calculate scores of given query tuple, on PAS counter object

        @takes:
            pasCounter:: colections.Counter
            pas_q :: (PRED, ARG0, ARG1)
        @returns:
            logscore:: log score of count
        """
        import math
        Logscore = lambda x, y: math.log(float(x) / float(y), 10) if x != 0 and y != 0 else 0
        try:
            count = pasCounter[pas_q]
        except KeyError:
            count = 10**(-6)
        logscore = Logscore(count, self.paslm_c_sum)
        return logscore

    def PASLM_count(self):
        self.invalidnum_plm = {"org":0, "alt":0}
        self.validnum_plm = {"org":0, "alt":0}
        for testid in self.case_keys:
            case = self.testcases[testid]
            self.testcases[testid]["PASLM_scores"] = {"org":[], "alt":[]}
            for org_pq in case["PASLM_queries"]["org"]:
                if org_pq[1] == "":
                    tmp = list(org_pq)
                    tmp.pop(1)
                    tmp.insert(1, "I")
                    org_pq = tuple(tmp)
                logging.debug(pformat(org_pq))
                try:
                    score = self._getPASLMscore(self.pasCounter, org_pq)
                    if score == 0:
                        self.invalidnum_plm["org"] += 1
                    else:
                        self.validnum_plm["org"] += 1
                    logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["PASLM_scores"]["org"].append(score)
            for alt_pq in case["PASLM_queries"]["alt"]:
                if alt_pq[1] == "":
                    tmp = list(alt_pq)
                    tmp.pop(1)
                    tmp.insert(1, "I")
                    alt_pq = tuple(tmp)
                logging.debug(pformat(alt_pq))
                try:
                    score = self._getPASLMscore(self.pasCounter, alt_pq)
                    if score == 0:
                        self.invalidnum_plm["alt"] += 1
                    else:
                        self.validnum_plm["alt"] += 1
                    logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["PASLM_scores"]["alt"].append(score)


    def _LM_PASLM_model(self, testcase, div_lm_pas=(1,5)):
        w_lm = div_lm_pas[0]
        w_pas = div_lm_pas[1]
        org_scores = [t for t in zip(testcase["LM_scores"]["org"], testcase["PASLM_scores"]["org"])]
        alt_scores = [t for t in zip(testcase["LM_scores"]["alt"], testcase["PASLM_scores"]["alt"])]
        detect_flag_lm = None # True if one of alternatives' score is over the original, None means there's no valid comparison
        detect_flag_pas = None
        if org_scores and alt_scores:
            for o_tuple in org_scores:
                o_s_lm, o_s_pas = o_tuple
                for a_tuple in alt_scores:
                    a_s_lm, a_s_pas = a_tuple
                    if a_s_lm > o_s_lm:
                        detect_flag_lm = True
                    elif o_s_lm > a_s_lm:
                        detect_flag_lm = False
                    if a_s_pas > o_s_pas:
                        detect_flag_pas = True
                    elif o_s_pas > a_s_pas:
                        detect_flag_pas = False
        if detect_flag_lm is True and detect_flag_pas is True:
            testcase["Result_LM+PASLM_model"] = "alt"
        elif detect_flag_lm is False and detect_flag_pas is True:
            if pasmodel_exclusive is True:
                testcase["Result_LM+PASLM_model"] = "alt"
            else:
                testcase["Result_LM+PASLM_model"] = "org"
        elif detect_flag_lm is True and detect_flag_pas is False:
            if pasmodel_exclusive is True:
                testcase["Result_LM+PASLM_model"] = "org"
            else:
                testcase["Result_LM+PASLM_model"] = "alt"
        elif detect_flag_lm is False and detect_flag_pas is False:
            testcase["Result_LM+PASLM_model"] = "org"
        elif detect_flag_lm is True and detect_flag_pas is None:
            testcase["Result_LM+PASLM_model"] = "alt"
        else:
            testcase["Result_LM+PASLM_model"] = "none_result"
        pass


    def _LM_model(self, testcase):
        """
        Compare original's score and alternatives' score
        then returns "org" or "alt" (in fce testset, "alt" is always right )
        """
        org_scores = [s for s in testcase["LM_scores"]["org"]]
        alt_scores = [s for s in testcase["LM_scores"]["alt"]]
        detect_flag = None # True if one of alternatives' score is over the original
        if org_scores and alt_scores:
            for o_s in org_scores:
                for a_s in alt_scores:
                    if a_s > o_s:
                        detect_flag = True
                    elif o_s > a_s:
                        detect_flag = False
        if detect_flag is True:
            testcase["Result_LM_model"] = "alt"
        elif detect_flag is False:
            testcase["Result_LM_model"] = "org"
        else:
            testcase["Result_LM_model"] = "failed"


    def _PASLM_model(self, testcase):
        org_scores = [s for s in testcase["PASLM_scores"]["org"]]
        alt_scores = [s for s in testcase["PASLM_scores"]["alt"]]
        detect_flag = None # True if one of alternatives' score is over the original
        if org_scores and alt_scores:
            for o_s in org_scores:
                for a_s in alt_scores:
                    if a_s > o_s:
                        detect_flag = True
                    elif o_s > a_s:
                        detect_flag = False
        if detect_flag is True:
            testcase["Result_PASLM_model"] = "alt"
        elif detect_flag is False:
            testcase["Result_PASLM_model"] = "org"
        else:
            testcase["Result_PASLM_model"] = "failed"


    def detect(self):
        for testid in self.case_keys:
            tc = self.testcases[testid]
            tc["Result_LM+PASLM_model"] = defaultdict(list)
            tc["Result_LM_model"] = defaultdict(list)
            tc["Result_PASLM_model"] = defaultdict(list)
            self._LM_model(tc)
            self._PASLM_model(tc)
            self._LM_PASLM_model(tc)
            logging.debug("Case %s"%(testid))
            logging.debug(pformat(tc))



def detectmain(corpuspath="", lmpath="", paslmpath="", reportout=""):
    detector = LM_Detector(corpuspath)
    detector.make_cases()
    detector.read_LM_and_PASLM(path_IRSTLM=lmpath, path_PASLM=paslmpath)
    if lmpath:
        detector.LM_count()
    if paslmpath:
        detector.PASLM_count()
    detector.detect()
    # detector.mk_report()





if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description =   """
                    Nyanco.detector

                    this detects verb replacement errors (RV) in FCE dataset,
                    using several methods.
                    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-p", "--pas_lm_path", action="store", 
                    help="path to PAS_LM (python collections.Counter object, as pickle)")
    ap.add_argument("-l", '--lm', action="store",
                    help="path to IRSTLM Language model file")
    ap.add_argument("-o", '--output_file', action="store",
                    help="path of output report file")
    ap.add_argument("-c", '--corpus_pickle_file', action="store",
                    help="path of pickled corpus made by corpusreader2.py and test_corpushandler.py")
    args = ap.parse_args()

    if (args.corpus_pickle_file and args.output_file and args.lm and args.pas_lm_path):
        print "Using both 5gramLM and PAS_triples"
        detectmain(corpus_dir=args.corpus_pickle_file, output_dir=args.output_file, working_dir=args.lm, preprocess=True)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    elif (args.corpus_pickle_file and args.output_file and args.lm):
        print "Using only 5gramLM"
        detectmain(corpus_dir=args.corpus_pickle_file, output_dir=args.output_file, working_dir=args.lm, preprocess=False)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    elif (args.corpus_pickle_file and args.output_file and args.pas_lm_path):
        print "Using only PAS_triples"
        detectmain(corpus_dir=args.corpus_pickle_file, output_dir=args.output_file, working_dir=args.lm, preprocess=False)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    else:
        ap.print_help()
    quit()