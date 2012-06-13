#! /usr/env/bin python
# coding: utf-8

'''
lexicalLoA.py

'''
import os, sys
from gensim import corpora, models, utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
from collections import defaultdict
# Nyanco/lsa_test/commonutils.py
import commonutils


class LexicalLoA(object):
    def __init__(self, dicpath, docpath, windowsize):
        self.dicpath = dicpath
        self.docpath = docpath
        self.N = int(windowsize)
        self.lexdic = corpora.Dictionary.load(self.dicpath)
        self.inputcorpus = open(self.docpath,'r').readlines()
        self.lexicalLoAdict = defaultdict(list) 


    def finddoc4words(self):
        for c, line in enumerate(self.inputcorpus):
            try:
                logging.debug('Processing line %d'%c)
                for word in self.lexdic.token2id.keys():
                    wordid = self.lexdic.token2id[word]
                    words = line.lower().strip('\n').split()
                    if word in words:
                        ngrams = commonutils.ngram(line, word, self.N)
                        for item in ngrams:
                            if item != []:
                                self.lexicalLoAdict[wordid].append(item)
            except KeyboardInterrupt:
                break
    

    def output(self):
        for id in self.lexicalLoAdict:
            if self.lexicalLoAdict[id]:
                filename = os.path.join('cooc/',self.lexdic[id] + '.coocdoc')
                with open(filename, 'w') as f:
                    for wlist in self.lexicalLoAdict[id]:
                        if len(wlist) < 1:
                            break
                        for word in set(wlist):
                            f.write(word.encode('utf-8')+' ')
                        f.write('\n')
        pickle.dump(self.lexicalLoAdict, open(os.path.join('cooc/LexicalLoA_window'+str(self.N)+'_afp_eng_2010.pkl'),'wb'))



def main():
    '''
    usage:

    $ lexicalLoA.py dictionaries/afp_eng_2010.gensimdict ./afp_eng_2010_preprocessed.txt 
    '''
    dicpath = sys.argv[1]
    docpath = sys.argv[2]
    windowsize = sys.argv[3]
    L = LexicalLoA(dicpath,docpath,windowsize)
    L.finddoc4words()
    L.output()


if __name__=='__main__':
    main()

