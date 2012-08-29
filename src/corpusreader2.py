#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/preprocessor2.py
Created on 29 Aug 2012
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "2.1"
__status__ = "Prototyping"


import os
import glob
import pickle
import json
from pprint import pformat
from datetime import datetime
logfilename = datetime.now().strftime("corpusreader2_log_%Y%m%d_%H%M.log")
# import copy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)
# in case lxml.etree isn't available...
try:
    from lxml import etree
    print("running with lxml.etree")
except ImportError:
    print("Import error")
    quit()
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def make_filelist(path="", prefix="", filetype=""):
    """
    @args
        path:: string of path to the directory
        prefix :: string of the prefix of files
        filetype:: stringf of the extension without a dot
    @returns 
        files:: a list of matched files in the path
    """
    import fnmatch
    if path and filetype:
        files = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, prefix + "*." + filetype):
                files.append(os.path.join(root,filename))
        return files
    else:
        return None



def store_to_pickle(path="", textlist=None):
    '''
    store obtained various grained sentence lists using pickles.
    
    '''
    if textlist:
        fname = path + datetime.now().strftime("_%Y%m%d_%H%M") + ".pickle"
        with open(fname, 'wb') as f:
            pickle.dump(textlist, f)
        return True
    else:
        return False


class CLCReader(object):
    '''
    Extracting error-tagged items from CLC-FCE dataset.
    xpath expressions are used to locate certain errors, mainly related verbs.
    '''
    
    def __init__(self, corpus_dir="", output_dir="", working_dir=""):
        if working_dir and corpus_dir:
            self.work_dir = working_dir
            self.corpus_dir = corpus_dir
            self.output_dir = output_dir
        else:
            raise IOError
        self.xmlfiles = make_filelist(path=self.corpus_dir, prefix="doc", filetype="xml")
        self.outputname = "FCE-dict"
        self.rawdata = []
        self.processed_sentences = []
        self.allincorrect_sents = []
        self.incorrect_verb_sents = []
        self.incorrect_RV_sents = []
        self.allcorrect_sents = []
        self.all_sents = []
        self.all_verberror = ["RV", "TV", "FV", "MV", "UV", "IV", "DV", "AGV"]
        self.RV_error = ["RV"]


    def read(self):
        '''
        A wrapper function

        @args
            none
        @returns
            all_sents :: flattened text as a list
            listindexdict :: a dictionary of {index<int>: filename<str>}
        '''
        self.all_sents = [correction_pairs for correction_pairs 
                          in [self.get_annotations(script) 
                              for script 
                              in [self.extract_script(etree.parse(script)) 
                                  for script in self.xmlfiles]]]
        self.listindexdict = dict((k,v) 
                                  for k,v 
                                  in zip(range(len(self.xmlfiles)), self.xmlfiles))
        return self.all_sents, self.listindexdict


    def extract_script(self, certain_etree):
        '''
        A method to obtain an element tree of exam-script of learner itself.
        returns listed script: ["some text", <NS element>, ...]
        '''
        try:
            script = certain_etree.xpath('//coded_answer//p/text()|.//NS')
            return(script)
        except:
            print("perhaps the given tree isn't actually a Element tree...")
            script = []
            return(script)


    def get_annotations(self, listed_tree):
        '''
        Retrieve the correction pairs (original, correction) of given 
        list of text elements and etree elements.
        The output will be;
        ['text', (org, corr), 'text', 'text', ...]
        '''
        corrpair = []
        for elem in listed_tree:
            try:
                corrpair.append(IfElem(elem).corrpair())
            except AttributeError:
                corrpair.append(IfStr(elem).corrpair())
        return(corrpair)


    def get_sentencelist(self, strlist):
        script = ""
        for wordchunk in strlist:
            try:
                if not wordchunk.endswith(" "):
                    script += wordchunk + " "
                elif not wordchunk.startswith(" "):
                    script += " " + wordchunk
                else:
                    script += wordchunk
            except:
                print "error on: ", wordchunk
        return(script)



class IfStr(object):
    '''
    This is an utility class for xmlreader.
    This can only print or return given object.
    '''
    def __init__(self, somestr):
        self.inputstring = unicode(somestr)
        
        
    def corrpair(self):
        return(self.inputstring)


class IfElem(object):
    '''
    This is an utility class for xmlreader.
    For etree.Element object, method 'corrpair' will obtain 
    a correction pair: (org, corr, $errortype)
        (when etree_element is <NS ***><i>foo</i><c>bar</c></NS>)
    '''
    
    def __init__(self, etree_element):
        self.elem = etree_element
        

    def check_errortype(self, etelem):
        try:
            errortype = etelem.values()[0]
            return(errortype)
        except IndexError, AttributeError:
            logging.debug(pformat("Couldn't get error type"))
            return(errortype)


    def corrpair(self):
        original = u""
        correction = u""
        try:
            errortype = unicode(self.check_errortype(self.elem))
        except:
            errortype = u""
        try:
            self.elem.text
        except AttributeError:
            raise AttributeError
        try:
            original = unicode(self.elem.find("i").text)
        except AttributeError:
            pass
        try:
            correction = unicode(self.elem.find("c").text)
        except AttributeError:
            pass
        correction_pair = (original,correction, errortype)
        if errortype == "RV":
            print pformat((original,correction,errortype))
        return(correction_pair)

# ===================================================================================================

# ===================================================================================================
class CLCPreprocessor(object):
    def __init__(self, xmllist, corpus):
        import nltk
        from nltk import sent_tokenize
        from nltk import word_tokenize
        self.corpus = [d for d in corpus]
        self.scripts = []
        self.annotations = []
        self.docs = xmllist


    def preprocess(self):
        docs = [d for d in self.corpus]
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
        sents = nltk.sent_tokenize(script)
        sents_tokenized = [nltk.wordpunct_tokenize(sent) for sent in sents]
        processed_sents = []
        annotations = [a for a in annotations_list]
        for sent in sents_tokenized:
            for index, word in enumerate(sent):
                if word == "{++}":
                    _annotation = annotations.pop(0)
                    len_i = len(_annotation[0])
                    len_c = len(_annotation[1])
                    sent.pop(index)
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
            processed_sents.append(sent)
        return processed_sents


    def retrieve(self, mode):
        docs = [doc for doc in self.scripts]
        annotations_list = [annotation for annotation in self.annotations]
        wordlist =  [ self._retrieve(script, annotations, mode) for (script, annotations) in zip(docs, annotations_list)]
        if mode == "Gold":
            self.goldwords = wordlist
        elif mode == "Incorrect":
            self.incorrwords = wordlist
        elif mode == "Incorrect_RV":
            self.incorr_RV = wordlist
        elif mode == "Incorrect_RV_check_main":
            self.incorr_RV_test_main = wordlist
        elif mode == "Incorrect_RV_check_correct":
            self.incorr_RV_test_correct = wordlist
        return True
    
    
    def _concat_words(self, wordlist):
        return reduce(lambda x, y: x + " " + y, wordlist)
     
     
    def output_raw(self, dest): 
        with open(dest, "w") as f:
            if "gold" in dest:
                for doc in self.goldwords:
                    for sent in doc:
                        s = self._concat_words(sent)
                        f.write(s.encode("utf-8") + "\n")
                    f.write("\n")
            elif "test" in dest:
                for doc in self.incorrwords:
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
                    docname = self.docs[i_doc]
                    for i_sent, (sent1, sent2) in enumerate(zip(doc1, doc2)):
                        s_main = self._concat_words(sent1)
                        s_correct = self._concat_words(sent2)
                        if "{incorrect!}" in s_main and "{correction}" in s_correct:
                            f.write("Doc: %s    Sentence: %d \n"%(docname[-24:], i_sent))
                            f.write(s_main.encode("utf-8") + "\n")
                            f.write(s_correct.encode("utf-8") + "\n")
                            f.write("\n")
       
       
def read(corpus_dir="", output_dir="", working_dir=""):
    C = CLCReader(corpus_dir=corpus_dir, output_dir=output_dir, working_dir=working_dir)
    C.read()
    return C.all_sents, C.listindexdict


def preprocess(xmllist, corpus_as_list, *args):
    processor = CLCPreprocessor(xmllist, corpus_as_list)
    processor.preprocess()
    for mode in args:
        processor.retrieve(mode)
#    print processor.goldwords[0:10]
#    print processor.incorrwords[0:10]
#    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-gold-raw_all.txt")
#    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-RVtest-last100docs.txt")
#    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-RV_check-all.txt")
    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-incorr-test_all.txt")
    processor.output_raw(dest)
    

def main(corpus_dir="", output_dir="", working_dir="", preprocess=False):
    corpus, filedict = read(corpus_dir=corpus_dir, output_dir=output_dir, working_dir=working_dir)


if __name__=='__main__':
    import time
    import sys
    import argparse
    starttime = time.time()
    argv = sys.argv
    argc = len(argv)
    description =   """This script will read and parse xml files of FCE dataset, 
                        and store them into python dictionary.
                    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-p", "--preprocess", action="store_true", 
                    help="put this if you need preprocessing for generating dictionary-formed corpus")
    ap.add_argument('-i', '--input_dir', action="store",
                    help="string of path to FCE corpus dir")
    ap.add_argument('-o', '--output_dir', action="store",
                    help="string of path-to-dir")
    ap.add_argument('-d', '--working_dir', action="store",
                    help="string of path working directory")
    args = ap.parse_args()
    if (args.input_dir and args.output_dir and args.working_dir and args.preprocess):
        main(corpus_dir=args.input_dir, output_dir=args.output_dir, working_dir=args.working_dir, preprocess=True)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    elif (args.input_dir and args.output_dir and args.working_dir):
        main(corpus_dir=args.input_dir, output_dir=args.output_dir, working_dir=args.working_dir, preprocess=False)
        endtime = time.time()
        print("\n\nOverall time %5.3f[sec.]"%(endtime - starttime))
    else:
        ap.print_help()
    quit()