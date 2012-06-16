#! /usr/env/bin python
# coding: utf-8
'''
extract_trainingtexts.py

This extract training text (raw) from given cooccurrence data
There're lots to be done...
'''
import os, sys
# from gensim import corpora, models, utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
from collections import defaultdict


class WordsFinder(object):
    def __init__(self, docpath):
        self.docpath = docpath
        self.inputcorpus = open(self.docpath,'r').readlines()
        self.wordsdict = defaultdict(list) 
        self.word2id = {'give':0, 'provide':1, 'offer':2, 'gave':0, 'given':0, 'provided':1, 'offered':2,
                'gives':0, 'provides':1, 'offers':2}
        self.id2word = defaultdict(list)
        for w, id in self.word2id.iteritems():
            self.id2word[id].append(w)
        print self.id2word.items()
    
    def _is_match(self, list1, list2):
        flag = False 
        for itemof1 in list1:
            if itemof1 in list2:
                flag = True
        return flag


    def finddoc4words(self):
        for c, line in enumerate(self.inputcorpus):
            logging.debug('Processing line %d'%c)
            for id, words in self.id2word.iteritems():
                if self._is_match(words, line.split()) is True:
                    self.wordsdict[id].append(line)
   

    def output(self):
        for id in self.wordsdict:
            if self.wordsdict[id]:
                word = self.id2word[id]
                filename = os.path.join('train/',str(id) + '.txt')
                sents = set(self.wordsdict[id])
                with open(filename, 'w') as f:
                    logging.debug('Writing file ID %s   name: %s'%(word, filename))
                    for line in sents:
                        f.write(line)
        pickle.dump(self.wordsdict, open(os.path.join('train/train_afp_eng_2009.pickle'),'w'))




# =============================================================================
# =============================================================================


def main():
    '''
    usage:

    $ python extract_trainingtexts.py afp_eng_2009_raw_clean.txt
    '''
    docpath = sys.argv[1]
    L = WordsFinder(docpath)
    L.finddoc4words()
    L.output()


if __name__=='__main__':
    main()

