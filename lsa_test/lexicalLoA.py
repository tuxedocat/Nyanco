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

class LexicalLoA(object):
    def __init__(self, dicpath, docpath):
        self.dicpath = dicpath
        self.docpath = docpath
        self.lexdic = corpora.Dictionary.load(self.dicpath)
        self.inputcorpus = open(self.docpath,'r').readlines()
        self.lexicalLoAdict = {}


    def finddoc4words(self):
        for c, line in enumerate(self.inputcorpus):
            logging.debug('Processing line %d'%c)
            for word in self.lexdic.token2id.keys():
                wordid = self.lexdic.token2id[word]
                if ' ' + word + ' ' in line:
                    if not wordid in self.lexicalLoAdict:
                        self.lexicalLoAdict[wordid] = [line]
                    else:
                        self.lexicalLoAdict[wordid].append(line)
    

    def output(self):
        for id in self.lexicalLoAdict:
            if self.lexicalLoAdict[id]:
                filename = os.path.join('cooc/',self.lexdic[id] + '.coocdoc')
                with open(filename, 'w') as f:
                    logging.debug('Writing file ID %d   name: %s'%(id, filename))
                    for line in self.lexicalLoAdict[id]:
                        f.write(line)
        pickle.dump(self.lexicalLoAdict, open(os.path.join('cooc/LexicalLoA_afp_eng_2010.pickle'),'w'))

def main():
    '''
    usage:

    $ lexicalLoA.py dictionaries/fce_all.gensimdict ./afp_eng_2010_preprocessed.txt
    '''
    dicpath = sys.argv[1]
    docpath = sys.argv[2]
    L = LexicalLoA(dicpath,docpath)
    L.finddoc4words()
    L.output()


if __name__=='__main__':
    main()

