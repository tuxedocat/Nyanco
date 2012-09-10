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
        self.docnames = sorted(self.corpus.keys())
        self.processedcorpus = defaultdict(dict)
        self.cp_tags = [u"RV",] # Extensible if tags other than RV are needed

    def _filter_checkpoints(self, dic_of_doc):
        """
        find checkpoints for a document, returns a filtered dictionary of the doc
        """
        self.checkpoints_of_doc = [] # index of "RV" errors' in the list
        filtereddict = defaultdict(list)
        for idx, ls in enumerate(dic_of_doc["errorposition"]):
            if ls:
                # print ls
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
        # logging.debug(pformat(filtereddict))
        return filtereddict

    def filter(self):
        for fn in self.docnames:
            # print self.corpus[fn].items()[0]
            self.processedcorpus[fn] = self._filter_checkpoints(self.corpus[fn])

    def parse_eachsents(self):
        """
        This creates temporary files of sentences, parse it, then retrieve it back to the dictionary
        """
        tmppath = os.path.join(os.path.dirname(self.path), "tmp")
        if not os.path.exists(tmppath):
            os.makedirs(tmppath)
        for name, doc in self.processedcorpus.iteritems():
            for idx, senttuple in enumerate(zip(doc["RVtest_text"], doc["gold_text"])): 
                with open(os.path.join(tmppath, name+"_test_part"+str(idx)+".tmp"), "w") as tf_t:
                    tf_t.write(senttuple[0])
                with open(os.path.join(tmppath, name+"_gold_part"+str(idx)+".tmp"), "w") as tf_g:
                    tf_g.write(senttuple[1])