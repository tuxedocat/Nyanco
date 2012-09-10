#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/preprocessor2.py
Created on 5 Sep. 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "2.0"
__status__ = "Prototyping"

import os
import nltk
from pprint import pformat
from collections import defaultdict
from datetime import datetime
logfilename = datetime.now().strftime("preprocessor2_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)

class CLCPreprocessor(object):
    """
    This class is for extracting corpus as a dictionary, with various criteria of error annotations

    The intermediate format is somehow like diff files
        "<some chunk> {++} <some chunk> <some chunk> {++}"
    pleces marked with {++} are corresponding to error tuples (<i>, <c>, <type>, <extra info>)

    """
    def __init__(self, corpus, filenamelist):
        self.corpus = corpus[0:] # copy elements in argument, create another object of the corpus
        self.scripts = []
        self.annotations = []
        self.docnamelist = [os.path.split(fn)[-1] for fn in filenamelist]
        self.corpusdict = {} # keys: filename, values: dictionaries which have attributes of the corpus


    def addmarkers(self):
        """
        Put markers {++} into raw sentences, and corresponding annotations into a list
        """
        # docs = [d for d in self.corpus]
        docs = self.corpus[0:]
        for doc in docs:
            entiredoc = ""
            annotations = [elem for elem in doc if isinstance(elem, tuple)]
            while doc:
                chunk = doc.pop(0)
                try:
                    entiredoc += chunk + " "
                except TypeError:
                    entiredoc += " {++} "
            self.scripts.append(entiredoc)
            self.annotations.append(annotations)


    def _retrieve(self, script, annotations_list, mode):
        """
        A function which is actually extracts annotations

        *_check_* modes will be omitted 
        """
        sents = nltk.sent_tokenize(script)
        sents_tokenized = [sent.split() for sent in sents]
        processed_sents = []
        annotations = [a for a in annotations_list]
        errorposition_list = []  # list of list of tuples (position, incorrect, correct, extra info.)
        for sent in sents_tokenized:
            tmplist_ep = []
            for index, word in enumerate(sent):
                if word == "{++}":
                    _annotation = annotations.pop(0)
                    len_i = len(_annotation[0])
                    len_c = len(_annotation[1])
                    sent.pop(index)
                    tmplist_ep.append((index, _annotation[0], _annotation[1], _annotation[2]))
                    if mode == "Gold":
                        _correct_word = _annotation[1]
                        sent.insert(index, _correct_word)
                    elif mode == "Incorrect_RV":
                        if _annotation[2] == "RV":
                            _replacement = _annotation[0]
                            sent.insert(index, _replacement)
                        else:
                            _replacement = _annotation[1]
                            sent.insert(index, _replacement)
                    elif mode == "Incorrect_RV_check_main":
                        if _annotation[2] == "RV":
                            if len_i > len_c:
                                _replacement = "{incorrect!}" + _annotation[0] + "{incorrect!}"
                            else:
                                _replacement = "{incorrect!}" + _annotation[0] + " "*(len_c-len_i) + "{incorrect!}"
                            sent.insert(index, _replacement)
                        else:
                            _replacement = _annotation[1]
                            sent.insert(index, _replacement)
                    elif mode == "Incorrect_RV_check_correct":
                        if _annotation[2] == "RV":
                            if len_c > len_i:
                                _replacement = "{correction}" + _annotation[1] + "{correction}"
                            else:
                                _replacement = "{correction}" + _annotation[1] + " "*(len_i-len_c) + "{correction}"
                            sent.insert(index, _replacement)
                        else:
                            _replacement = _annotation[1]
                            sent.insert(index, _replacement)
                    elif mode == "Incorrect":
                        _replacement = _annotation[0]
                        sent.insert(index, _replacement)
            errorposition_list.append(tmplist_ep)
            processed_sents.append(sent)
        return processed_sents, errorposition_list


    def retrieve(self, mode):
        """
        A wrapper function for retrieving annotations
        """
        # docs = [doc for doc in self.scripts]
        docs = self.scripts[0:]
        # annotations_list = [annotation for annotation in self.annotations]
        annotations_list = self.annotations[0:]
        # list_of_words = [ self._retrieve(script, annotations, mode) for (script, annotations) in zip(docs, annotations_list)]
        list_of_words = []
        list_of_errorposition = []
        for (script, annotations) in zip(docs, annotations_list):
            words, errorposition = self._retrieve(script, annotations, mode)
            list_of_words.append(words)
            list_of_errorposition.append(errorposition)
        self.errorposition_list = list_of_errorposition[0:]
        if mode == "Gold":
            self.goldwords = list_of_words[0:]
        elif mode == "Incorrect":
            self.incorrwords = list_of_words[0:]
        elif mode == "Incorrect_RV":
            self.incorr_RV = list_of_words[0:]
        elif mode == "Incorrect_RV_check_main":
            self.incorr_RV_test_main = list_of_words[0:]
        elif mode == "Incorrect_RV_check_correct":
            self.incorr_RV_test_correct = list_of_words[0]
        return True


    def create_corpus(self):
        self.dict_corpus = defaultdict(dict) 
        for docname in self.docnamelist:
            self.dict_corpus[docname] = defaultdict(list)

        for idx, script in enumerate(self.goldwords):
            docname = self.docnamelist[idx]
            self.dict_corpus[docname]["gold_words"] = script
            self.dict_corpus[docname]["gold_text"] = []
            for sent in script:
                cats = self._concat_words(sent)
                if cats:
                    self.dict_corpus[docname]["gold_text"].append(cats)
                else:
                    self.dict_corpus[docname]["gold_text"].append("")

        for idx, script in enumerate(self.incorr_RV):
            docname = self.docnamelist[idx]
            self.dict_corpus[docname]["RVtest_words"] = script
            self.dict_corpus[docname]["RVtest_text"] = []
            for sent in script:
                cats = self._concat_words(sent)
                if cats:
                    self.dict_corpus[docname]["RVtest_text"].append(cats)
                else:
                    self.dict_corpus[docname]["RVtest_text"].append("")

        for idx, errorlist in enumerate(self.errorposition_list):
            docname = self.docnamelist[idx]
            self.dict_corpus[docname]["errorposition"] = errorlist
        return self.dict_corpus

    
    def _concat_words(self, wordlist):
        if wordlist != [] and wordlist != [[]]:
            return u" ".join(wordlist)
        else:
            return None
     
    def output_raw(self, destlist): 
        for dest in destlist: 
            with open(dest, "w") as f:
                if "gold" in dest:
                    for doc in self.goldwords:
                        for sent in doc:
                            s = self._concat_words(sent)
                            f.write(s.encode("utf-8") + "\n")
                        f.write("\n")
                elif "test" in dest:
                    for doc in self.incorr_RV:
                        for sent in doc:
                            s = self._concat_words(sent)
                            f.write(s.encode("utf-8") + "\n")
                        f.write("\n")
                elif "RVtest" in dest:
                    for doc in self.incorr_RV[-99:]:
                        for sent in doc:
                            s = self._concat_words(sent)
                            f.write(s.encode("utf-8") + "\n")
                        f.write("\n")
                elif "RV_check" in dest:
                    for i_doc, (doc1,doc2) in enumerate(zip(self.incorr_RV_test_main, self.incorr_RV_test_correct)):
                        docname = self.docnamelist[i_doc]
                        for i_sent, (sent1, sent2) in enumerate(zip(doc1, doc2)):
                            s_main = self._concat_words(sent1)
                            s_correct = self._concat_words(sent2)
                            if "{incorrect!}" in s_main and "{correction}" in s_correct:
                                f.write("Doc: %s    Sentence: %d \n"%(docname[-24:], i_sent))
                                f.write(s_main.encode("utf-8") + "\n")
                                f.write(s_correct.encode("utf-8") + "\n")
                                f.write("\n")


def preprocess(corpus_as_list=[], filenamelist=[], modelist=[], destlist=[]):
    """
    Wrapper function of preprocessor2

    @ARGS:
        corpus_as_list:: [[chunks and annotation-tuples], [each list is corresponded to a document], 
                            [existing file is the same as its index in filenamelist]]
        filenamelist:: a list of filenames
        modelist:: list of modes in CLCPreprocessor
            "Gold", "Incorrect_RV", "Incorrect", "Incorrect_RV_check_main", "Incorrect_RV_check_correct" 

    @TODO:
        * take args as a list by argparse
        * destlist will be moved to corpusreader2
    """
    pp = CLCPreprocessor(corpus_as_list, filenamelist)
    pp.addmarkers()
    for mode in modelist:
        pp.retrieve(mode)
    # pp.output_raw(destlist)
    corpus = pp.create_corpus()
    return corpus
    # print pp.goldwords
    # destlist = [os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-gold-raw_all.txt"),
    #             os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-RVtest-last100docs.txt"),
    #             os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-RV_check-all.txt"),
    #             os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-incorr-test_all.txt")]
    # pp.output_raw(destlist)

