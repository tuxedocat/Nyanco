# ! /usr/bin/env python
# coding: utf-8

from nose.plugins.attrib import attr
from feature_extractor import SimpleFeatureExtractor
from datetime import datetime
import logging
logfilename = datetime.now().strftime("detector_log_%Y%m%d_%H%M.log")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='../log/'+logfilename)
import os
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
from numpy import array
from pattern.text import en
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction import DictVectorizer


class TestFext(object):
    def setUp(self):
        self.corpuspath = "../sandbox/classify/out"
        self.testpath = "../sandbox/classify/FEtest/test.pkl2"
        # self.corpus = SimpleFeatureExtractor.read_corpusfiles(self.corpuspath)
        # self.verbset = pickle.load(open("../tool/verbset_111_20.pkl2", "rb"))
        # self.checkdata = defaultdict(list)
        # for key in self.verbset["verbs"][:10]:
        #     self.checkdata[key] = pickle.load(open(os.path.join(self.corpuspath, "%s.pkl2"%key), "rb"))
        self.testdata = pickle.load(open(self.testpath,"rb"))
        self.dat_single =  ['1\tChinese\t_\t_\tJJ\t_\t3\tamod\t_\t_\t_\t_\t_\t_',
                            '2\therbal\t_\t_\tJJ\t_\t3\tamod\t_\t_\t_\t_\t_\t_',
                            '3\tmedicines\t_\t_\tNNS\t_\t4\tnsubj\t_\t_\t_\t_\tARG0\t5',
                            '4\tmay\t_\t_\tMD\t_\t0\tROOT\t_\t_\t_\t_\tARGM-MOD\t5',
                            '5\twork\t_\t_\tVB\t_\t4\tvch\t_\t_\twork.01\t_\t_\t_',
                            '6\tbecause\t_\t_\tIN\t_\t5\tprep\t_\t_\t_\t_\tARGM-CAU\t5',
                            '7\tof\t_\t_\tIN\t_\t6\tcombo\t_\t_\t_\t_\t_\t_',
                            '8\tthe\t_\t_\tDT\t_\t9\tdet\t_\t_\t_\t_\t_\t_',
                            '9\tadulterants.\t_\t_\tNN\t_\t6\tpobj\t_\t_\t_\t_\t_\t_']
        self.fe = SimpleFeatureExtractor(self.dat_single, "work")
    @attr("feature_simple")
    def test_single(self):
        self.fe.ngrams(n=5, v_pos=4)
        ngramfeature = []
        for i in range(5):
            ngramfeature.append(self.fe.features)
        logging.debug(pformat(ngramfeature))
        vec = DictVectorizer(sparse=True)
        array_f = vec.fit_transform(ngramfeature).toarray()
        logging.debug(pformat(array_f))

        raise Exception