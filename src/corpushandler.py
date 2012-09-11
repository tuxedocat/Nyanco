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

# ----------------------------------------------------------------------------------------------------

class CorpusHandler(object):
    def __init__(self, corpuspath):
        """
        Constructor args:
            corpuspath: path of a pickled corpus created by corpusreader2.py
            e.g. "/Users/foo/bar/fce.pickle"
        """
        try:
            cf = pickle.load(open(corpuspath, "rb"))
            self.corpus = cf
        except IOError:
            print "File doesn't exist"
            raise IOError
        self.path = corpuspath
        self.tmppath = os.path.join(os.path.dirname(self.path), "tmp")
        self.parsedpath = os.path.join(os.path.dirname(self.path), "parsed")
        self.docnames = sorted(self.corpus.keys())
        self.processedcorpus = defaultdict(dict)
        self.cp_tags = [u"RV",] # Extensible if tags other than RV are needed


    def main(self):
        self.filter()
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
                        filtereddict["errorposition"].append(et)
                        filtereddict["RVtest_words"].append(dic_of_doc["RVtest_words"][idx])
                        filtereddict["RVtest_text"].append(dic_of_doc["RVtest_text"][idx])
                        filtereddict["gold_words"].append(dic_of_doc["gold_words"][idx])
                        filtereddict["gold_text"].append(dic_of_doc["gold_text"][idx])
                    else:
                        pass
        return filtereddict

    def filter(self):
        for fn in self.docnames:
            self.processedcorpus[fn] = self._filter_checkpoints(self.corpus[fn])

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
                    doc["RVtest_tags"] = rawtags
                with open(fn_g, "r") as tf_g:
                    rawtags = [tuple(t.strip("\n").split("\t")) for t in tf_g.readlines() if t != "\n"]
                    doc["gold_tags"] = rawtags
                pe_t = pas_extractor.PEmod(fn_t)
                pe_g = pas_extractor.PEmod(fn_g)
                doc["RVtest_PAS"] = pe_t.extract_full()
                doc["gold_PAS"] = pe_g.extract_full()
                print pformat(doc["RVtest_PAS"])
                print pformat(doc["gold_PAS"])
        with open(os.path.join(os.path.dirname(self.path), "fce_processed.pickle"), "wb") as cf:
            pickle.dump(self.processedcorpus, cf)
        return True