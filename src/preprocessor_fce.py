#! /usr/bin/python
# coding: utf-8
'''
Nyanco/preprocessor_fce.py
Created on 4 May 2012

This will read FCE released dataset XMLs and output them as python list object.
For stanford-coreNLP POS/Dependency annotator, this class have simple sentence
processing feature using NLTK.

@author: Yu Sawai
@version: 0.001
'''

import os
import glob
import pickle
import json
import nltk


# in case lxml.etree isn't available...
try:
    from lxml import etree
    print("running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        print("running with cElementTree on Python 2.5+")
    except ImportError:
        print("Import error")


class CLCReader(object):
    '''
    Extracting error-tagged items from CLC-FCE dataset.
    xpath expressions are used to locate certain errors, mainly related verbs.
                    
    TODO: hand extracted sentences to preprocessor 
    TODO: make a version for NUCLE corpus, BNC 
    '''
    
    def __init__(self):
        self.tagset = []
        try:
            self.workingdir = os.path.join(os.environ["WORKDIR"], 'Nyancodat/')
            print "WORKINGDIR: ", self.workingdir
        except:
            self.workingdir = os.path.join(os.environ["HOME"], 'cl/Nyancodat/')
            print "WORKINGDIR: ", self.workingdir
        try:
            self.CORPUS_DIR_ROOT = os.path.join(os.environ["WORKDIR"], 'cl/nldata/')
            print "Reading corpus from: ", self.CORPUS_DIR_ROOT
        except:
            self.CORPUS_DIR_ROOT = os.path.join(os.environ["HOME"], 'cl/nldata/')
            print "Reading corpus from: ", self.CORPUS_DIR_ROOT
        self.rawdata = []
        self.allxml = CorpusFileHandler(self.CORPUS_DIR_ROOT, "clc").mklist()
        self.processed_sentences = []
        self.allincorrect_sents = []
        self.incorrect_verb_sents = []
        self.incorrect_RV_sents = []
        self.allcorrect_sents = []
        self.all_sents = []
        self.verberrors_all = ["RV", "TV", "FV", "MV", "UV", "IV", "DV", "AGV"]
        self.verberrors_specific = ["RV"]
        self.collocationerrors = ["CL"]
        self.outputname = "extracted.ertx"


    def read(self):
        '''
        A wrapper function
        '''
        self.all_sents = [correction_pairs for correction_pairs 
                          in [self.get_annotations(script) 
                              for script 
                              in [self.extract_script(etree.parse(script)) 
                                  for script in self.allxml]]]
        self.listindexdict = dict((k,v) 
                                  for k,v 
                                  in zip(range(len(self.allxml)), self.allxml))
        return self.all_sents, self.listindexdict


    def store_to_pickle(self, textlist):
        '''
        store obtained various grained sentence lists using pickles.
        
        '''
        with open(self.workingdir + self.outputname, 'wb') as f:
            pickle.dump(textlist, f)


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
        errortype = u""
        self.elem = etree_element
        

    def check_errortype(self, etelem):
        try:
            errortype = etelem.values()[0]
            return(errortype)
        except IndexError, AttributeError:
            print "can't get errortype"
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
        return(correction_pair)



class CorpusFileHandler(object):
    '''
    This class gives a simple interface to access corpora.
    Absolute path to the parent directory to the corpora is needed.
    corpora_parent_dir = /home/hoge/somedir/     (NUCLE, CLC,...)
    '''
    def __init__(self, corpus_parent_dir, corpus_name):
        import os
        self.CORPORA = {"clc":os.path.join(corpus_parent_dir,"fce-released-dataset/dataset/"),
                        "nucle":os.path.join(corpus_parent_dir,"NUCLE/shaped/")}
        self.filelist = []
        self.corpus_dir = ''
        try:
            self.corpus_dir = self.CORPORA[corpus_name]
        except KeyError:
            print("Not found. %s"%corpus_name)

    def mklist(self):
        try:
            self.filelist = glob.glob(self.corpus_dir+"*/*.xml")
            filelist = self.filelist
        except:
            print("Errors occured in mklist")
        return filelist


class CLCPreprocessor(object):
    def __init__(self, corpus):
        import nltk
        from nltk import sent_tokenize
        from nltk import word_tokenize
        self.corpus = [d for d in corpus]
        self.scripts = []
        self.annotations = []


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
#            print sent
            for index, word in enumerate(sent):
#                print word
                if word == "{++}":
#                    print annotations
                    _annotation = annotations.pop(0)
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
        return True
    
    
    def _concat_words(self, wordlist):
        return reduce(lambda x, y: x + " " + y, wordlist)

     
     
    def output_raw(self, dest): 
        with open(dest, "w+") as f:
            if "gold" in dest:
                for doc in self.goldwords[0:-100]:
                    for sent in doc:
                        s = self._concat_words(sent)
                        f.write(s.encode("utf-8") + "\n")
                    f.write("\n")
            elif "test" in dest:
                for doc in self.incorrwords[-99:]:
                    for sent in doc:
                        s = self._concat_words(sent)
                        f.write(s.encode("utf-8") + "\n")
                    f.write("\n")
            elif "RVtest" in dest:
                for doc in self.incorrwords[-99:]:
                    for sent in doc:
                        s = self._concat_words(sent)
                        f.write(s.encode("utf-8") + "\n")
                    f.write("\n")
                    
                    
class NUCLEReader(object):
    def __init__(self):
        self.tagset = []
        self.CORPUS_PATH_RELATIVE = 'usr/share/NUCLE/shaped'
        self.DATAPATH = os.path.join(os.environ['HOME'], self.CORPUS_PATH_RELATIVE)
    


def read():
    C = CLCReader()
    C.read()
    print(C.all_sents[0:10])
    print(C.listindexdict)
    return C.all_sents


def preprocess(corpus_as_list, *args):
    processor = CLCPreprocessor(corpus_as_list)
    processor.preprocess()
    for mode in args:
        processor.retrieve(mode)
    print processor.goldwords[0:10]
    print processor.incorrwords[0:10]
#    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-gold-raw.txt")
    dest = os.path.join(os.environ['HOME'], "cl/FCE-processed/fce-RVtest-last100docs.txt")
    processor.output_raw(dest)
    
    

if __name__ == "__main__":
    import time
    total =time.time()
    corpus = read()
    preprocess(corpus, "Gold", "Incorrect", "Incorrect_RV")
    
    endtime = time.time()
    print("\n\nOverall time %5.3f[sec.]"%(endtime - total))