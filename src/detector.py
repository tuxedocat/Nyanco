# ! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/detector.py
Created on 9 Sep. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os, glob
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from pattern.text import en
from nltk.corpus import wordnet as wn
try:
    from lsa_test.irstlm import *
except:
    from tool.irstlm_moc import *
import tool.altword_generator as altgen
from classifier import BoltClassifier
from sklearn.datasets import svmlight_format
from sklearn import cross_validation
import numpy as np
import bolt
from feature_extractor import SimpleFeatureExtractor
from tool.sparse_matrices import *


class DetectorBase(object):
    def __init__(self, corpusdictpath="", reportpath="", verbsetpath=""):
        if os.path.exists(corpusdictpath):
            with open(corpusdictpath, "rb") as f:
                corpusdict = pickle.load(f)
                self.corpus = corpusdict
            self.experimentset = defaultdict(dict)
        else:
            raise IOError
        self.reportpath = os.path.join(os.path.dirname(reportpath), 
                                        datetime.now().strftime("detector_report_%Y%m%d_%H%M.log"))
        reportdir = os.path.dirname(self.reportpath)
        if not os.path.exists(reportdir):
            os.makedirs(reportdir)
        self.ngram_len = 5
        self.verbsetpath = verbsetpath
        self.altreader = altgen.AlternativeReader(self.verbsetpath)
        self.verbset = pickle.load(open(self.verbsetpath, "rb"))


    def make_cases(self):
        """
        An alternative version of make_cases.
        This takes a pickled corpus of {"checkpoints_RV":<corpus as dict>, "checkpoints_VB":<corpus as dict>}
        Then put two test-cases together, with additional key "has_checkpoints":<True or False>
        """
        self.testcases = defaultdict(dict)
        self.case_keys = []
        self.dataset_with_cp = self.corpus["checkpoints_RV"]
        self.dataset_without_cp = self.corpus["checkpoints_VB"]
        for docname, doc in self.dataset_with_cp.iteritems():
            try:
                self._mk_cases(docname=docname, doc=doc, is_withCP=True)
            except KeyError as ke:
                logging.debug(pformat(ke))
            except Exception as e:
                logging.debug("error catched in make_cases for checkpoints")
                logging.debug(pformat(e))
                raise

        for docname, doc in self.dataset_without_cp.iteritems():
            try:
                if docname in self.testcases.keys():
                    docname_dup = docname + "2"
                    self._mk_cases(docname=docname_dup, doc=doc, is_withCP=False)
                else:
                    self._mk_cases(docname=docname, doc=doc, is_withCP=False)
            except KeyError as ke:
                logging.debug(pformat(ke))
            except Exception as e:
                logging.debug("error catched in make_cases for other sentences")
                logging.debug(docname)
                logging.debug(pformat(e))
                raise


    def detect(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class SupervisedDetector(DetectorBase):
    """
    Supervised (multiclass classification) based detector implementation
    Currently, methods such as _mk_cases is just copied and did some modifications from the original
    TODO:
        Better, smart implementation for shared codes such as _mk_cases
    """
    def readmodels(self, path_dataset_root="", modeltype="sgd", toolkit="sklearn"):
        dirlist = glob.glob(os.path.join(path_dataset_root, "*"))
        namelist = [os.path.basename(p) for p in dirlist]
        self.verb2modelpath = {vn : p for (vn, p) in zip(namelist, dirlist)}
        self.models = {}
        self.fmaps = {}
        self.label2id = {}
        self.tempdir = os.path.dirname(path_dataset_root)
        self.toolkit = toolkit
        if os.path.basename(self.tempdir) == "dataset":
            self.tempdir = os.path.join(path_dataset_root, os.pardir)
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)
        for setname, modelroot in self.verb2modelpath.iteritems():
            try:
                with open(os.path.join(modelroot,"model_"+modeltype+".pkl2"), "rb") as mf:
                    self.models[setname] = pickle.load(mf)
            except:
                self.models[setname] = None
            with open(os.path.join(modelroot,"featuremap.pkl2"), "rb") as mf:
                self.fmaps[setname] = pickle.load(mf)
            with open(os.path.join(modelroot,"label2id.pkl2"), "rb") as mf:
                self.label2id[setname] = pickle.load(mf)
        if self.toolkit == "sklearn":
            self.datapath = [os.path.join(self.tempdir, "X.npz"), os.path.join(self.tempdir, "Y.npy")]
        elif self.toolkit == "bolt":
            self.datapath = os.path.join(self.tempdir, "dataset.svmlight")


    def __addcheckpoints(self, doc=None):
        cp_list = []
        for tag_for_sent in doc["gold_tags"]:
            vblist = [(idx, tag[1]) for idx, tag in enumerate(tag_for_sent) if "VB" in tag[2] and tag[5] != "be.01"]
            # print pformat(tag_for_sent)
            if vblist:
                cp_list.append([(tuple[0], tuple[1], tuple[1], "NoError") for tuple in vblist])
            else:
                cp_list.append([])
        # print "Checkpoints_VB: ", pformat(cp_list)
        return cp_list


    def _mk_cases(self, docname="", doc=None, is_withCP=True):
        if docname and doc:
            try:
                gold_tags = doc["gold_tags"]; test_tags = doc["RVtest_tags"]
                gold_text = doc["gold_text"]; test_text = doc["RVtest_text"]
                gold_words = doc["gold_words"]; test_words = doc["RVtest_words"]
                gold_pas = doc["gold_PAS"]; test_pas = doc["RVtest_PAS"]
                if is_withCP is True:
                    checkpoints = doc["errorposition"]
                    for cpid, cp in enumerate(checkpoints):
                        testkey = docname + "_checkpoint_RV_" + str(cpid)
                        cp_pos = cp[0]
                        incorr = cp[1]
                        gold = cp[2]
                        test_wl = test_words[cpid]
                        self.testcases[testkey]["gold_text"] = gold_text[cpid]
                        self.testcases[testkey]["test_text"] = test_text[cpid]
                        self.testcases[testkey]["checkpoint_idx"] = cp_pos
                        self.testcases[testkey]["incorrect_label"] = altgen.AlternativeReader.get_lemma(incorr)
                        self.testcases[testkey]["gold_label"] = altgen.AlternativeReader.get_lemma(gold)
                        self.testcases[testkey]["type"] = "RV"
                        self.testcases[testkey]["features"] = mk_features(tags=test_tags[cpid], v=incorr)
                        self.case_keys.append(testkey)
                else:
                    checkpoints = self.__addcheckpoints(doc)
                    for s_id, sent_cp in enumerate(checkpoints):
                        if sent_cp:
                            for cpid, cp in enumerate(sent_cp):
                                if cp:
                                    testkey = docname + "_checkpoint_VB_" + str(s_id) + "." + str(cpid)
                                    cp_pos = cp[0]
                                    incorr = cp[1]
                                    gold = cp[2]
                                    test_wl = test_words[s_id]
                                    self.testcases[testkey]["gold_text"] = gold_text[s_id]
                                    self.testcases[testkey]["test_text"] = test_text[s_id]
                                    self.testcases[testkey]["gold_words"] = gold_words[s_id]
                                    self.testcases[testkey]["test_words"] = test_words[s_id]
                                    self.testcases[testkey]["checkpoint_idx"] = cp_pos
                                    self.testcases[testkey]["incorrect_label"] = altgen.AlternativeReader.get_lemma(incorr)
                                    self.testcases[testkey]["gold_label"] = altgen.AlternativeReader.get_lemma(gold)
                                    self.testcases[testkey]["type"] = ""
                                    self.testcases[testkey]["features"] = mk_features(tags=test_tags[s_id], v=incorr)
                                    self.case_keys.append(testkey)
            except Exception, e:
                logging.debug("error catched in _mk_cases")
                raise
                # logging.debug(pformat(docname))
                # logging.debug(pformat(doc))
                # logging.debug(pformat(e))

    def _bolt_pred(self, model=None):
        if model:
            test = bolt.io.MemoryDataset.load(self.datapath)
            X = test.instances
            Y = [p for p in model.predict(X)]
            if Y:
                return Y[0]

    def _sklearn_pred(self, model=None, X=None, Y=None):
        if model:
            return model.predict(X)

    def _sklearn_pred_prob(self, model=None, X=None, Y=None):
        if model:
            return model.predict_prob(X)[0]

    def get_classification(self):
        """
        calculate scores of given instances
        """
        for testid in self.case_keys:
            case = self.testcases[testid]
            strfeature = case["features"]
            setname = case["incorrect_label"]
            y = case["gold_label"]
            try:
                model = self.models[setname]
                fmap = self.fmaps[setname]
                lmap = self.label2id[setname]
                logging.debug("SupervisedDetector: model for %s is found :)"%setname)
                case["incorr_classid"] = lmap[setname]
                case["is_incorr_in_Vset"] = True
            except KeyError:
                logging.debug("SupervisedDetector: model for %s is not found :("%setname)
                model = None
                fmap = None
                lmap = None
                case["incorr_classid"] = None
                case["is_incorr_in_Vset"] = False
            try:
                classid = lmap[y]
                case["gold_classid"] = classid
                case["is_gold_in_Vset"] = True
            except (KeyError, TypeError):
                classid = -100
                case["gold_classid"] = classid
                case["is_gold_in_Vset"] = False
            if model and fmap:
                _X = fmap.transform(strfeature)
                _Y = np.array([classid])
                if self.toolkit == "bolt":
                    with open(self.datapath+"temp", "wb") as f:
                        svmlight_format.dump_svmlight_file(_X, _Y, f, comment=None)
                    with open(self.datapath+"temp", "rb") as f:
                        cleaned = f.readlines()[2:]
                    with open(self.datapath, "wb") as f:
                        f.writelines(cleaned)
                        os.remove(self.datapath+"temp")
                    output = self._bolt_pred(model)
                elif self.toolkit == "sklearn":
                    output = self._sklearn_pred(model, _X, _Y)
                    print self._sklearn_pred_prob(model, _X, _Y)
                case["classifier_output"] = output
            else:
                case["classifier_output"] = -100


    def _check(self, true_error, org, cls_out):
        if true_error:
            if org == cls_out:
                return 0
            else:
                return 1
        else:
            if org == cls_out:
                return 0
            else:
                return 1


    def mk_report(self):
        """
        Classes:
            0: Not a verb error
            1: Verb error
        """
        from sklearn import metrics
        self.truelabels = []
        self.syslabels = []
        self.gold_in_Cset = []
        labels = [0 ,1]
        names = ["not_verb-error", "verb-error"]
        for id, case in self.testcases.iteritems():
            try:
                gold = case["gold_classid"]
                org = case["incorr_classid"]
                cls_out = case["classifier_output"]
                tmp_l = []
                assert case["is_incorr_in_Vset"] == True
                if case["type"] == "RV":
                    self.syslabels.append(self._check(True, org, cls_out))
                    self.truelabels.append(1)
                else:
                    self.syslabels.append(self._check(False, org, cls_out))
                    self.truelabels.append(0)
                if case["is_gold_in_Vset"] == True:
                    self.gold_in_Cset.append(1)
            except Exception, e:
                logging.debug(pformat(e))

        with open(self.reportpath, "w") as rf:
            ytrue = np.array(self.truelabels)
            ysys = np.array(self.syslabels)
            print ytrue
            print ysys
            # skf = cross_validation.StratifiedKFold(ytrue, k=5)
            # for tridx, teidx in skf:
            #     _ytrue = ytrue[teidx]
            #     _ysys = ysys[teidx]
            #     clsrepo_lm = metrics.classification_report(_ytrue, _ysys, target_names=names)
            #     cm_lm = metrics.confusion_matrix(_ytrue, _ysys, labels=np.array([0,1]))
            #     print clsrepo_lm
            #     print pformat(cm_lm)
            #     rf.write(clsrepo_lm)
            #     rf.write("\n\n")
            clsrepo_lm = metrics.classification_report(np.array(ytrue), np.array(ysys), target_names=names)
            cm_lm = metrics.confusion_matrix(np.array(ytrue), np.array(ysys), labels=np.array([0,1]))
            print clsrepo_lm
            print pformat(cm_lm)
            print "num. of [words in FCE-gold which are covered by Cset] case", len(self.gold_in_Cset)
            rf.write(clsrepo_lm)


def mk_features(tags=[], v=""):
    fe = SimpleFeatureExtractor(tags=tags, verb=v)
    fe.ngrams(n=7)
    # some more features are needed
    return fe.features


def detectmain_c(corpuspath="", model_root="", type="sgd", reportout="", verbsetpath=""):
    try:
        detector = SupervisedDetector(corpusdictpath=corpuspath, verbsetpath=verbsetpath, reportpath=reportout)
        detector.make_cases()
        detector.readmodels(path_dataset_root=model_root, modeltype=type)
        detector.get_classification()
        detector.mk_report()
    except Exception, e:
        print pformat(e)
        raise

#-------------------------------------------------------------------------------
# LM models
#-------------------------------------------------------------------------------

class LM_Detector(DetectorBase):
    def read_LM_and_PASLM(self, path_IRSTLM="", path_PASLM=""):
        if path_IRSTLM:
            self.LM = initLM(5, path_IRSTLM)
            logging.debug(pformat("IRSTLM's LM is loaded from %s"%path_IRSTLM))
        if path_PASLM:
            self.pasCounter = pickle.load(open(path_PASLM))
            logging.debug(pformat("PASLM is loaded"))
            self.paslm_c_sum = sum(self.pasCounter.values())

    def cleanup(self):
        print "Deleting LM...."
        deleteLM(self.LM)
        logging.debug("IRSTLM_LM has been deleted from memory")
        print "Deleting LM has been completed" 


    def _mk_ngram_queries(self, n=5, cp_pos=None, w_list=[], alt_candidates=[]):
        """
        Make a query for ngram frequency counter
        @takes:
            n :: N gram size (if n=5, [-2 -1 word +1 +2])
            cp_pos:: int, positional index of the checkpoint
            w_list:: list, words of a sentence
            alt_candidates:: list, alternative candidates if given
        @returns:
            org_q:: list of string, queries for irstlm.getSentenceScore (Original word)
            alt_q:: list of string, queries for irstlm.getSentenceScore (Generated by given candidates)
            TODO: moc_smart_alt_q:: list of string, ad hoc moc of Smartquery
        """
        org_q = []
        alt_q = []
        window = int((n - 1)/2)
        core = w_list[cp_pos]
        _left = [word for index, word in enumerate(w_list) if index < cp_pos][-window:]
        _right = [word for index, word in enumerate(w_list) if index > cp_pos][:window]
        org_q = [str(" ".join(_left)) + " "+ core + " " + str(" ".join(_right))]
        for alt in alt_candidates:
            alt_q.append(str(" ".join(_left)) + " "+ alt + " " + str(" ".join(_right)))
        return org_q, alt_q


    def _mk_PAS_queries(self, pasdiclist=[], org_preds=[], alt_preds=[]):
        org_pas_q = []
        alt_pas_q = []
        try:
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
        except:
            pass
        # if org_pas_q and alt_pas_q:
            # logging.debug(pformat(org_pas_q))
            # logging.debug(pformat(alt_pas_q))
        return org_pas_q, alt_pas_q


    def __addcheckpoints(self, doc=None):
        cp_list = []
        for tag_for_sent in doc["gold_tags"]:
            vblist = [(idx, tag[1]) for idx, tag in enumerate(tag_for_sent) if "VB" in tag[2] and tag[5] != "be.01"]
            # print pformat(tag_for_sent)
            if vblist:
                cp_list.append([(tuple[0], tuple[1], tuple[1], "NoError") for tuple in vblist])
            else:
                cp_list.append([])
        # print "Checkpoints_VB: ", pformat(cp_list)
        return cp_list


    def __read_altwords(self, orgword=""):
        return self.altreader.get_altwordlist(orgword)


    def _mk_cases(self, docname="", doc=None, is_withCP=True):
        if docname and doc:
            try:
                gold_tags = doc["gold_tags"]; test_tags = doc["RVtest_tags"]
                gold_text = doc["gold_text"]; test_text = doc["RVtest_text"]
                gold_words = doc["gold_words"]; test_words = doc["RVtest_words"]
                gold_pas = doc["gold_PAS"]; test_pas = doc["RVtest_PAS"]
                if is_withCP is True:
                    checkpoints = doc["errorposition"]
                    for cpid, cp in enumerate(checkpoints):
                        testkey = docname + "_checkpoint_RV_" + str(cpid)
                        cp_pos = cp[0]
                        incorr = cp[1]
                        gold = cp[2]
                        test_wl = test_words[cpid]
                        self.testcases[testkey]["gold_text"] = gold_text[cpid]
                        self.testcases[testkey]["test_text"] = test_text[cpid]
                        self.testcases[testkey]["checkpoint_idx"] = cp_pos
                        self.testcases[testkey]["incorrect_label"] = altgen.AlternativeReader.get_lemma(incorr)
                        self.testcases[testkey]["gold_label"] = altgen.AlternativeReader.get_lemma(gold)
                        query_altwords = self.__read_altwords(altgen.AlternativeReader.get_lemma(incorr))
                        org_qs, alt_qs = self._mk_ngram_queries(n=self.ngram_len, cp_pos=cp_pos, w_list=test_wl, alt_candidates=query_altwords)
                        self.testcases[testkey]["LM_queries"] = {"org":org_qs, "alt":alt_qs}
                        org_pqs, alt_pqs = self._mk_PAS_queries(pasdiclist=gold_pas+test_pas, org_preds=[incorr], alt_preds=query_altwords)
                        self.testcases[testkey]["PASLM_queries"] = {"org":org_pqs, "alt":alt_pqs}
                        self.testcases[testkey]["type"] = "RV"
                        self.case_keys.append(testkey)
                else:
                    checkpoints = self.__addcheckpoints(doc)
                    # print pformat(checkpoints)
                    for s_id, sent_cp in enumerate(checkpoints):
                        # print pformat(("sent_id %d, checkpoints :"%s_id, sent_cp))
                        # print "sent_gold", str(gold_words)
                        # print "sent_test", str(test_words)
                        if sent_cp:
                            for cpid, cp in enumerate(sent_cp):
                                # print pformat(("cpid %d, checkpoints :"%cpid, cp))
                                if cp:
                                    testkey = docname + "_checkpoint_VB_" + str(s_id) + "." + str(cpid)
                                    cp_pos = cp[0]
                                    incorr = cp[1]
                                    gold = cp[2]
                                    test_wl = test_words[s_id]
                                    query_altwords = self.__read_altwords(altgen.AlternativeReader.get_lemma(incorr))
                                    self.testcases[testkey]["gold_text"] = gold_text[s_id]
                                    self.testcases[testkey]["test_text"] = test_text[s_id]
                                    self.testcases[testkey]["gold_words"] = gold_words[s_id]
                                    self.testcases[testkey]["test_words"] = test_words[s_id]
                                    self.testcases[testkey]["checkpoint_idx"] = cp_pos
                                    self.testcases[testkey]["incorrect_label"] = altgen.AlternativeReader.get_lemma(incorr)
                                    self.testcases[testkey]["gold_label"] = altgen.AlternativeReader.get_lemma(gold)
                                    org_qs, alt_qs = self._mk_ngram_queries(n=self.ngram_len, cp_pos=cp_pos, w_list=test_wl, alt_candidates=query_altwords)
                                    self.testcases[testkey]["LM_queries"] = {"org":org_qs, "alt":alt_qs}
                                    org_pqs, alt_pqs = self._mk_PAS_queries(pasdiclist=gold_pas+test_pas, org_preds=[incorr], alt_preds=query_altwords)
                                    self.testcases[testkey]["PASLM_queries"] = {"org":org_pqs, "alt":alt_pqs}
                                    self.testcases[testkey]["type"] = ""
                                    self.case_keys.append(testkey)

            except Exception, e:
                logging.debug("error catched in _mk_cases")
                # logging.debug(pformat(docname))
                # logging.debug(pformat(doc))
                # logging.debug(pformat(e))


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


    def _LM_model(self, testcase):
        """
        Compare original's score and alternatives' score
        then returns "org" or "alt"
        """
        org_scores = [s for s in testcase["LM_scores"]["org"]]
        alt_scores = [s for s in testcase["LM_scores"]["alt"]]
        detect_flag = None # True if one of alternatives' score is over the original
        if org_scores and alt_scores:
            for o_s in org_scores:
                for a_s in alt_scores:
                    if a_s > o_s:
                        detect_flag = True
                    elif o_s >= a_s:
                        detect_flag = False
        if detect_flag is True:
            testcase["Result_LM_model"] = "alt"
        elif detect_flag is False:
            testcase["Result_LM_model"] = "org"
        elif detect_flag is None:
            testcase["Result_LM_model"] = "none_result"

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
                # logging.debug(pformat(org_pq))
                try:
                    score = self._getPASLMscore(self.pasCounter, org_pq)
                    if score == 0:
                        self.invalidnum_plm["org"] += 1
                    else:
                        self.validnum_plm["org"] += 1
                    # logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["PASLM_scores"]["org"].append(score)
            for alt_pq in case["PASLM_queries"]["alt"]:
                if alt_pq[1] == "":
                    tmp = list(alt_pq)
                    tmp.pop(1)
                    tmp.insert(1, "I")
                    alt_pq = tuple(tmp)
                # logging.debug(pformat(alt_pq))
                try:
                    score = self._getPASLMscore(self.pasCounter, alt_pq)
                    if score == 0:
                        self.invalidnum_plm["alt"] += 1
                    else:
                        self.validnum_plm["alt"] += 1
                    # logging.debug(pformat(score))
                except TypeError:
                    score = -100
                self.testcases[testid]["PASLM_scores"]["alt"].append(score)


    def _LM_PASLM_model(self, testcase, pasmodel_exclusive=True):
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
                    elif o_s_lm >= a_s_lm:
                        detect_flag_lm = False
                   
                    if a_s_pas > o_s_pas:
                        detect_flag_pas = True
                    elif o_s_pas >= a_s_pas:
                        detect_flag_pas = False
        elif testcase["LM_scores"]["org"] and testcase["LM_scores"]["alt"]:
            o_scores = testcase["LM_scores"]["org"]
            a_scores = testcase["LM_scores"]["alt"]
            for o_s in o_scores:
                for a_s in a_scores:
                    if a_s > o_s:
                        detect_flag_lm = True
                    elif o_s >= a_s:
                        detect_flag_lm = False

        if detect_flag_lm is True and detect_flag_pas is True:
            testcase["Result_LMPASLM_model"] = "alt"
        elif detect_flag_lm is False and detect_flag_pas is True:
            if pasmodel_exclusive is True:
                testcase["Result_LMPASLM_model"] = "alt"
            else:
                testcase["Result_LMPASLM_model"] = "org"
        elif detect_flag_lm is True and detect_flag_pas is False:
            if pasmodel_exclusive is True:
                testcase["Result_LMPASLM_model"] = "org"
            else:
                testcase["Result_LMPASLM_model"] = "alt"
        elif detect_flag_lm is False and detect_flag_pas is False:
            testcase["Result_LMPASLM_model"] = "org"
        elif detect_flag_lm is True and detect_flag_pas is None:
            testcase["Result_LMPASLM_model"] = "alt"
        elif detect_flag_lm is False and detect_flag_pas is None:
            testcase["Result_LMPASLM_model"] = "org"
        elif detect_flag_lm is None and detect_flag_pas is True:
            testcase["Result_LMPASLM_model"] = "alt"
        elif detect_flag_lm is None and detect_flag_pas is False:
            testcase["Result_LMPASLM_model"] = "org"
        else:
            testcase["Result_LMPASLM_model"] = "none_result"
        pass


    def _PASLM_model(self, testcase):
        org_scores = [s for s in testcase["PASLM_scores"]["org"]]
        alt_scores = [s for s in testcase["PASLM_scores"]["alt"]]
        detect_flag = None # True if one of alternatives' score is over the original
        if org_scores and alt_scores:
            for o_s in org_scores:
                for a_s in alt_scores:
                    if a_s > o_s:
                        detect_flag = True
                    elif o_s >= a_s:
                        detect_flag = False
        if detect_flag is True:
            testcase["Result_PASLM_model"] = "alt"
        elif detect_flag is False:
            testcase["Result_PASLM_model"] = "org"
        elif detect_flag is None:
            testcase["Result_PASLM_model"] = "none_result"


    def detect(self):
        for testid in self.case_keys:
            tc = self.testcases[testid]
            tc["Result_LM_model"] = defaultdict(list)
            # tc["Result_PASLM_model"] = defaultdict(list)
            # tc["Result_LMPASLM_model"] = defaultdict(list)
            self._LM_model(tc)
            # self._PASLM_model(tc)
            # self._LM_PASLM_model(tc)
            # logging.debug("Case %s"%(testid))
            # logging.debug(pformat(tc))


    def mk_report(self):
        """
        Classes:
            0: Not a verb error
            1: Verb error
        """
        from sklearn import metrics
        self.truelabels = []
        self.syslabels_lm_paslm = []
        self.syslabels_lm = []
        self.syslabels_paslm = []
        self.report = [] 
        labels = [0 ,1]
        names = ["not_verb-error", "verb-error"]
        for id, case in self.testcases.iteritems():
            try:
                tmpdic_r = {}
                truelabel = case["gold_label"]
                incorrlabel = case["incorrect_label"]
                # if "Result_LMPASLM_model" in case:
                #     lm_paslm_out = case["Result_LMPASLM_model"]
                # if "Result_LM_model" in case:
                lm_out = case["Result_LM_model"]
                # if "Result_PASLM_model" in case:
                #     paslm_out = case["Result_PASLM_model"]
                # output_models = [lm_paslm_out, lm_out, paslm_out]
                output_models = [lm_out]
                # print "LMout", lm_out
                tmp_l = []
                for output in output_models:
                    if output == "alt":
                        tmp_l.append(1)
                    elif output == "none_result":
                        tmp_l.append(0)
                    elif output == "org":
                        tmp_l.append(0)
                if case["type"] == "RV":
                    self.syslabels_lm.append(tmp_l[0])
                    self.truelabels.append(1)
                else:
                    if "org" in output_models:
                        self.syslabels_lm.append(0)
                        self.truelabels.append(0)
                    else:
                        self.syslabels_lm.append(tmp_l[0])
                        self.truelabels.append(0)
                # self.syslabels_lm_paslm.append(tmp_l[0])
                # self.syslabels_lm.append(tmp_l[1])
                # self.syslabels_paslm.append(tmp_l[2])
                tmpdic_r["name"] = id
                tmpdic_r["outputs"] = output_models
                tmpdic_r["original"] = incorrlabel
                tmpdic_r["correction"] = truelabel
                self.report.append(tmpdic_r)
            except Exception, e:
                print pformat(e)
                print pformat(case)

        with open(self.reportpath, "w") as rf:
            ytrue = np.array(self.truelabels)
            ysys = np.array(self.syslabels_lm)
            # skf = cross_validation.StratifiedKFold(ytrue, k=5)
            # for tridx, teidx in skf:
            #     _ytrue = ytrue[teidx]
            #     _ysys = ysys[teidx]
            #     clsrepo_lm = metrics.classification_report(_ytrue, _ysys, target_names=names)#, labels=labels, target_names=names)
            #     cm_lm = metrics.confusion_matrix(_ytrue, _ysys, labels=np.array([0,1]))
            #     print clsrepo_lm
            #     print pformat(cm_lm)
            #     rf.write(clsrepo_lm)
            #     rf.write("\n\n")
            # clsrepo_lm_paslm = metrics.classification_report(np.array(self.truelabels), np.array(self.syslabels_lm_paslm), target_names=names)#, labels=labels, target_names=names)
            # clsrepo_paslm = metrics.classification_report(np.array(self.truelabels), np.array(self.syslabels_paslm), target_names=names)#, labels=labels, target_names=names)
            # print clsrepo_lm_paslm
            # clsrepo_lm = metrics.classification_report(np.array(self.truelabels), np.array(self.syslabels_lm), target_names=names)#, labels=labels, target_names=names)
            
            cm_lm = metrics.confusion_matrix(np.array(self.truelabels), np.array(self.syslabels_lm), labels=np.array([0,1]))
            # print clsrepo_lm
            rf.write("\n\n")
            # rf.write(clsrepo_lm)
            print pformat(cm_lm)
            # try:
            #     for repo in self.report:
            #         rf.write(pformat(repo))
            # except Exception as fileouterror:
            #     logging.debug(pformat(fileouterror))


def detectmain(corpuspath="", lmpath="", paslmpath="", reportout="", verbsetpath=""):
    try:
        detector = LM_Detector(corpusdictpath=corpuspath, reportpath=reportout, verbsetpath=verbsetpath)
        detector.make_cases()
        detector.read_LM_and_PASLM(path_IRSTLM=lmpath, path_PASLM=paslmpath)
        if lmpath:
            detector.LM_count()
        if paslmpath:
            detector.PASLM_count()
        detector.detect()
        detector.mk_report()
    except Exception, e:
        print pformat(e)
    finally :
        detector.cleanup()


if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description =   """
                    Nyanco.detector\nthis detects verb replacement errors (RV) in FCE dataset,using several methods.

                    \n\npython detector.py -M classifier -t sgd -m ../sandbox/classify/tiny/datasets -v ../sandbox/classify/verbset_tiny.pkl2 -c ../sandbox/fce_corpus/fce_dataset_v2_tiny.pkl2 -o ../log/classifier_detector_test"""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-p", "--pas_lm_path", action="store", 
                    help="path to PAS_LM (python collections.Counter object, as pickle)")
    ap.add_argument("-l", '--lm', action="store",
                    help="path to IRSTLM Language model file")
    ap.add_argument("-o", '--output_file', action="store",
                    help="path of output report file")
    ap.add_argument("-c", '--corpus_pickle_file', action="store",
                    help="path of pickled corpus made by corpusreader2.py and test_corpushandler.py")
    ap.add_argument("-M", '--model', action="store",
                    help="classifier or lm")
    ap.add_argument("-t", '--model_type', action="store",
                    help="sgd | pegasos | pa")
    ap.add_argument("-m", '--model_dir_root', action="store",
                    help="root path of dataset/model directory")
    ap.add_argument("-v", '--verbset', action="store",
                    help="root path of verbset pickle")
    args = ap.parse_args()

    if (args.corpus_pickle_file and args.output_file and args.lm and args.pas_lm_path):
        print "Using both 5gramLM and PAS_triples"
        detectmain(corpuspath=args.corpus_pickle_file, reportout=args.output_file, lmpath=args.lm, paslmpath=args.pas_lm_path, verbsetpath=args.verbset)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    elif (args.corpus_pickle_file and args.output_file and args.lm and args.model=="lm"):
        print "Using only 5gramLM"
        detectmain(corpuspath=args.corpus_pickle_file, reportout=args.output_file, lmpath=args.lm, paslmpath=args.pas_lm_path, verbsetpath=args.verbset)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    elif (args.corpus_pickle_file and args.output_file and args.pas_lm_path):
        print "Using only PAS_triples"
        detectmain(corpuspath=args.corpus_pickle_file, reportout=args.output_file, lmpath=args.lm, paslmpath=args.pas_lm_path, verbsetpath=args.verbset)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))

    elif (args.corpus_pickle_file and args.output_file and args.model=='classifier'):
        print "Using Classifier models"
        detectmain_c(corpuspath=args.corpus_pickle_file, reportout=args.output_file, verbsetpath=args.verbset, model_root=args.model_dir_root, type=args.model_type)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))


    else:
        ap.print_help()
    quit()
