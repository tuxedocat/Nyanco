#! /usr/env/bin python
# coding: utf-8
import os, sys
from gensim import corpora, models, utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle


class WordsFinder(object):
    def __init__(self, docpath):
        self.docpath = docpath
        self.inputcorpus = open(self.docpath,'r').readlines()
        self.wordsdict = {}
        self.words=['learn', 'know', 'study', 'learnt', 'learned', 'studied', 'known', 'knew', 'read']

    def finddoc4words(self):
        for c, line in enumerate(self.inputcorpus):
            logging.debug('Processing line %d'%c)
            for id, word in enumerate(self.words):
                if ' ' + word + ' ' in line:
                    if not word in self.wordsdict:
                        self.wordsdict[word] = [line]
                    else:
                        self.wordsdict[word].append(line)
    

    def output(self):
        for id in self.wordsdict:
            if self.wordsdict[id]:
                filename = os.path.join('train/',id + '.txt')
                sents = set(self.wordsdict[id])
                with open(filename, 'w') as f:
                    logging.debug('Writing file ID %s   name: %s'%(id, filename))
                    for line in sents:
                        f.write(line)
        pickle.dump(self.wordsdict, open(os.path.join('train/train_afp_eng_2009.pickle'),'w'))

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

