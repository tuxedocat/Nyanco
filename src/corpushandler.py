#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/corpushandler.py
Created on 9 Sep 2012

This reads pickled corpus, find checkpoint, count frequency or ...

LOG:
    9 Sep 2012: first commit

'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = ".0"
__status__ = "Prototyping"


import os
import cPickle as pickle
import json
from pprint import pformat
from datetime import datetime
logfilename = datetime.now().strftime("handler_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)
from collections import defaultdict
from tool import pas_extractor
from tool.online_fanseparser import OnlineFanseParser

# ----------------------------------------------------------------------------------------------------

class CorpusHandler(object):
    def __init__(self, corpuspath="", outputname=""):
        """
        Constructor args:
            corpuspath: path of a pickled corpus created by corpusreader2.py
            e.g. "/Users/foo/bar/fce.pickle"

            outputname: name of processedcorpus
        """
        try:
            cf = pickle.load(open(corpuspath, "rb"))
            self.corpus = cf
        except IOError:
            print "File doesn't exist"
            raise IOError
        self.path = corpuspath
        self.tmppath = os.path.join(os.path.dirname(self.path), "tmp")
        self.tmppath_op = os.path.join(os.path.dirname(self.path), "fptmp")
        self.parsedpath = os.path.join(os.path.dirname(self.path), "parsed")
        if not os.path.exists(self.tmppath):
            os.makedirs(self.tmppath)
        if not os.path.exists(self.tmppath_op):
            os.makedirs(self.tmppath_op)
        if not os.path.exists(self.parsedpath):
            os.makedirs(self.parsedpath)
        self.docnames = sorted(self.corpus.keys())
        self.processedcorpus = defaultdict(dict)
        self.processedcorpus_others = defaultdict(dict)
        self.cp_tags = [u"RV",] # Extensible if tags other than RV are needed
        self.outputname = outputname


    def main(self, mode=""):
        if mode == "checkpoints":
            self.filter_checkpoints()
        elif mode == "others":
            self.filter_others()

        if not (os.path.exists(self.tmppath) and len(self.processedcorpus.keys()) == len(os.listdir(self.tmppath))):
            print "creating pre-parsing data..."
            self.parse_eachsent_pre()
        if (os.path.exists(self.parsedpath) and len(self.processedcorpus.keys()) == len(os.listdir(self.parsedpath))):
            print "reading post-parsing data"
            self.parse_eachsent_read()
        else:
            print "parse .tmp files with fanseparser first!!!!"


    def _filter_checkpoints(self, dic_of_doc):
        """
        find checkpoints for a document, returns a filtered dictionary of the doc
        """
        self.checkpoints_of_doc = [] # index of "RV" errors' in the list
        filtereddict = defaultdict(list)
        for idx, ls in enumerate(dic_of_doc["errorposition"]):
            if ls:
                for et in ls:
                    flag = et[3]
                    if flag in self.cp_tags:
                        logging.debug(pformat((idx, et)))
                        filtereddict["errorposition"].append(et)
                        filtereddict["RVtest_words"].append(dic_of_doc["RVtest_words"][idx])
                        filtereddict["RVtest_text"].append(dic_of_doc["RVtest_text"][idx])
                        filtereddict["gold_words"].append(dic_of_doc["gold_words"][idx])
                        filtereddict["gold_text"].append(dic_of_doc["gold_text"][idx])
                    else:
                        pass
        return filtereddict

    def filter_checkpoints(self):
        for fn in self.docnames:
            self.processedcorpus[fn] = self._filter_checkpoints(self.corpus[fn])


    def _filter_others(self, dic_of_doc):
        """
        find checkpoints for a document, exclude sentences contain checkpoints 
        """
        filtereddict = defaultdict(list)
        for idx, ls in enumerate(dic_of_doc["errorposition"]):
            if ls:
                RV = [et for et in ls if et[3] in self.cp_tags]
                if not RV:
                    # logging.debug(pformat((idx, dic_of_doc["RVtest_text"][idx])))
                    filtereddict["errorposition"].append(())
                    filtereddict["RVtest_words"].append(dic_of_doc["RVtest_words"][idx])
                    filtereddict["RVtest_text"].append(dic_of_doc["RVtest_text"][idx])
                    filtereddict["gold_words"].append(dic_of_doc["gold_words"][idx])
                    filtereddict["gold_text"].append(dic_of_doc["gold_text"][idx])
                else:
                    pass
            else:
                # logging.debug(pformat((idx, dic_of_doc["RVtest_text"][idx])))
                filtereddict["errorposition"].append(())
                filtereddict["RVtest_words"].append(dic_of_doc["RVtest_words"][idx])
                filtereddict["RVtest_text"].append(dic_of_doc["RVtest_text"][idx])
                filtereddict["gold_words"].append(dic_of_doc["gold_words"][idx])
                filtereddict["gold_text"].append(dic_of_doc["gold_text"][idx])
        return filtereddict

    def filter_others(self):
        for fn in self.docnames:
            self.processedcorpus_others[fn] = self._filter_others(self.corpus[fn])


    def onlineparse_others(self):
        ofp = OnlineFanseParser(w_dir=self.tmppath_op)
        ofp.check_running()
        try:
            n_all = len(self.processedcorpus_others)
            c = 0
            for name, doc in self.processedcorpus_others.iteritems():
                c += 1
                for idx, sent in enumerate(doc["gold_text"]): 
                    print "Parsing and getting PAS tags... (%i of %i)"%(c, n_all)
                    logging.debug(pformat("Parsing and getting PAS tags... (%i of %i)"%(c, n_all)))
                    parsed = ofp.parse_one(sent)
                    parsed_t = [tuple(l.split("\t")) for l in parsed]
                    doc["gold_tags"].append(parsed_t)
                    doc["RVtest_tags"].append(parsed_t)
                    pe = pas_extractor.OnlinePasExtractor(parsed)
                    doc["RVtest_PAS"].append(pe.extract_full())
                    doc["gold_PAS"].append(pe.extract_full())
                    logging.debug(pformat(doc["RVtest_PAS"]))
                    logging.debug(pformat(doc["gold_tags"]))
        except KeyboardInterrupt:
            print "Interrupted... aborting parsing."
            pass

        finally:
            # ofp.clean()
            pass

        if self.outputname == "":
            outputname = "fce_processed_v2VB.pickle3"
        else:
            outputname = self.outputname
        with open(os.path.join(os.path.dirname(self.path), outputname), "wb") as cf:
            pickle.dump(self.processedcorpus_others, cf, -1)
        return True

    def onlineparse_RV(self):
        ofp = OnlineFanseParser(w_dir=self.tmppath_op)
        ofp.check_running()
        try:
            n_all = len(self.processedcorpus)
            c = 0
            for name, doc in self.processedcorpus.iteritems():
                c += 1
                for idx, sent in enumerate(doc["gold_text"]): 
                    print "Parsing and getting PAS tags... (%i of %i)"%(c, n_all)
                    logging.debug(pformat("Parsing and getting PAS tags... (%i of %i)"%(c, n_all)))
                    print pformat(sent)
                    print pformat(doc["RVtest_text"][idx])
                    # try:
                    g_parsed = ofp.parse_one(sent)
                    g_parsed_t = [tuple(l.split("\t")) for l in g_parsed]
                    doc["gold_tags"].append(g_parsed_t)
                    t_parsed = ofp.parse_one(doc["RVtest_text"][idx])
                    t_parsed_t = [tuple(l.split("\t")) for l in t_parsed]
                    doc["RVtest_tags"].append(t_parsed_t)
                    g_pe = pas_extractor.OnlinePasExtractor(g_parsed)
                    t_pe = pas_extractor.OnlinePasExtractor(t_parsed)
                    doc["RVtest_PAS"].append(g_pe.extract_full())
                    doc["gold_PAS"].append(t_pe.extract_full())
                        # logging.debug(pformat(doc["RVtest_PAS"]))
                        # logging.debug(pformat(doc["gold_tags"]))
                    # except TypeError, e:
                    #     print pformat(e)
                    #     print "error occured in ", doc["RVtest_text"]
                    # except Exception as oe:
                    #     print pformat(oe)
                    #     print "error occured in ", doc["RVtest_text"]
            
        except KeyboardInterrupt:
            print "Interrupted... aborting parsing."
            pass
        except Exception, e:
            print pformat(e)
            raise
        finally:
            # ofp.clean()
            pass

        if self.outputname == "":
            outputname = "fce_processed_v2RV.pickle3"
        else:
            outputname = self.outputname
        with open(os.path.join(os.path.dirname(self.path), outputname), "wb") as cf:
            pickle.dump(self.processedcorpus, cf, -1)
        return True

# ----------------------------------------------------------------------
# NOTE: following code will be useless since now fanseparser-mod is used
# ----------------------------------------------------------------------

    def parse_eachsent_pre(self):
        """
        This creates temporary files of sentences, parse it (currently, this is done manually by shell command)
        """
        if not os.path.exists(self.tmppath):
            os.makedirs(self.tmppath)
        if not os.path.exists(self.parsedpath):
            os.makedirs(self.parsedpath)
        for name, doc in self.processedcorpus.iteritems():
            for idx, senttuple in enumerate(zip(doc["RVtest_text"], doc["gold_text"])): 
                with open(os.path.join(self.tmppath, name+"_test_part"+str(idx)+".tmp"), "w") as tf_t:
                    tf_t.write(senttuple[0])
                with open(os.path.join(self.tmppath, name+"_gold_part"+str(idx)+".tmp"), "w") as tf_g:
                    tf_g.write(senttuple[1])
        return True

    def parse_eachsent_read(self):
        """
        Once the tempfiles created, this function reads parsed infomation into the corpus-dictionary

        Assuming .tmp files have already been parsed by fanseparser, and put in './parsed' dirs
        """
        for name, doc in self.processedcorpus.iteritems():
            for idx, senttuple in enumerate(zip(doc["RVtest_text"], doc["gold_text"])): 
                fn_t = os.path.join(self.parsedpath, name+"_test_part"+str(idx)+".parsed")
                fn_g = os.path.join(self.parsedpath, name+"_gold_part"+str(idx)+".parsed")
                with open(fn_t, "r") as tf_t:
                    rawtags = [tuple(t.strip("\n").split("\t")) for t in tf_t.readlines() if t != "\n"]
                    # doc["RVtest_tags"] = rawtags
                    doc["RVtest_tags"].append(rawtags)
                with open(fn_g, "r") as tf_g:
                    rawtags = [tuple(t.strip("\n").split("\t")) for t in tf_g.readlines() if t != "\n"]
                    # doc["gold_tags"] = rawtags
                    doc["gold_tags"].append(rawtags)
                pe_t = pas_extractor.PEmod(fn_t)
                pe_g = pas_extractor.PEmod(fn_g)
                # doc["RVtest_PAS"] = pe_t.extract_full()
                doc["RVtest_PAS"].append(pe_t.extract_full())
                # doc["gold_PAS"] = pe_g.extract_full()
                doc["gold_PAS"].append(pe_g.extract_full())
                # print pformat(doc["RVtest_PAS"])
                # print pformat(doc["gold_PAS"])
        if self.outputname == "":
            outputname = "fce_processed_2.pickle"
        else:
            outputname = self.outputname
        with open(os.path.join(os.path.dirname(self.path), outputname), "wb") as cf:
            pickle.dump(self.processedcorpus, cf)
        return True