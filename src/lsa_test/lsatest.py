# /usr/bin/env python
# coding: utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import gensim
from gensim import corpora, models, utils, similarities
import re
import os
import glob
import sys


class Corpus(object):
    def __init__(self, filename):
        corpusfile = open(filename, 'r')

    def __iter__(self):
        for line in corpusfile:
            yield dictionary.doc2bow(line.lower().split())


class LexicalLoA_CoocMatrix(object):
    def __init__(self, row_vocabpath, column_vocabpath, coocdirpath):
        self.row_vocabpath = row_vocabpath
        self.column_vocabpath = column_vocabpath
        self.coocdoc_path = coocdirpath
        self.row_vocabdic =  corpora.Dictionary.load(self.row_vocabpath)
        self.row_vocabdic.items();
        self.column_vocabdic = corpora.Dictionary.load(self.column_vocabpath)
        self.column_vocabdic.items();
        self.stopwords = [w.strip('\n') for w in open('stopwords.txt','r').readlines()]
        self.CORPUSNAME = 'afp_eng_2010'


    def convert2bowcorpus(self):
        corpus_raw = []
        corpus_bow = []
        for id, word in self.column_vocabdic.id2token.iteritems():
            print os.path.join(self.coocdoc_path,word+'.coocdoc')
            try:
                coocfilestring = open(os.path.join(self.coocdoc_path,word+'.coocdoc'), 'r').read()
                words = utils.simple_preprocess(coocfilestring.replace('\n', ' '))
                words = [w for w in words if w not in self.stopwords]
                corpus_raw.append(words)
                corpus_bow.append(self.column_vocabdic.doc2bow(words))
            except IOError:
                corpus_raw.append([' '])
                corpus_bow.append(self.column_vocabdic.doc2bow([' ']))
        self.corpus = corpus_bow
        corpora.SvmLightCorpus.serialize(os.path.join('corpora/',self.CORPUSNAME+'_LexicalLoA.svmlight'), corpus_bow)

def main():
    '''
    $python ./lsatest.py ./dictionaries/afp_eng_2010.gensimdict ./dictionaries/afp_eng_2010.gensimdict ./cooc
    '''
    row_dicpath = sys.argv[1]
    column_dicpath = sys.argv[2]
    coocdir_path = sys.argv[3]
    L = LexicalLoA_CoocMatrix(row_dicpath,column_dicpath,coocdir_path)
    L.convert2bowcorpus()


if __name__=='__main__':
    main()
