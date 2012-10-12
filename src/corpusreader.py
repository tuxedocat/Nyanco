#! /usr/bin/python
# coding: utf-8
'''
Nyanco/corpusreader.py
Created on 30 Dec 2011

NOTE:
    This will be deleted since now corpusreader2 is used

@author: Yu Sawai
@version: 0.001
'''

import os
import glob
import pickle
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
        self.errortypes = ["AllOriginal", "IncorrectVerbs", 
                           "IncorrectRV", "AllCorrect", "AllInfo"]
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
        return self.all_sents
        
        
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


    def read_extract_all(self, *args):
        for examscript in self.allxml:
            temp_tree = etree.parse(examscript)
            temp_script = self.extract_script(temp_tree)
            temp_corrpairs = self.get_annotations(temp_script)
            if "AllOriginal" in args:
                self.allincorrect_sents.append(self.get_strlist
                                               (temp_corrpairs, "AllOriginal"))
            if "IncorrectVerbs" in args:
                self.incorrect_verb_sents.append(self.get_strlist
                                                 (temp_corrpairs, "IncorrectVerbs"))
            if "IncorrectRV" in args:
                self.incorrect_RV_sents.append(self.get_strlist
                                               (temp_corrpairs, "IncorrectRV"))
            if "AllCorrect" in args:
                self.allcorrect_sents.append(self.get_strlist
                                             (temp_corrpairs, "AllCorrect"))


    def get_strlist(self, corrpairlist, errortype=""):
        '''
        ARGS:
            corrpairlist: the output list of get_annotations
            errortype: "AllOriginal", "IncorrectVerbs", 
                       "IncorrectRV", "AllCorrect"
        RETURNS:
            incorr: a list represents error script.
        '''
        
        strlist = []
        for elem in corrpairlist:
            try:
                elem.islower            # if elem is string...
                strlist.append(elem)
            except AttributeError:
                print(elem)
                if errortype == "AllOriginal":
                    try:
                        strlist.append("<error>"+elem[0].strip()+"</error>")
                    except AttributeError:
                        strlist.append("<error>"+"</error>")
                elif errortype == "AllCorrect":
                    strlist.append(elem[1])
                elif errortype == "IncorrectVerbs":
                    if elem[2] in self.verberrors_all:
                        try:
                            strlist.append("<VE>"+elem[0].strip()+"</VE>")
                        except AttributeError:
                            strlist.append("<VE>"+"</VE>")
                    else:
                        strlist.append(elem[1])
                elif errortype == "IncorrectRV":
                    if elem[2] in self.verberrors_specific:
                        try:
                            strlist.append("<RV>"+elem[0].strip()+"</RV>")
                        except AttributeError:
                            strlist.append("<RV>"+"</RV>")
                    else:
                        strlist.append(elem[1])
                elif errortype == "AllInfo":
                    strlist.append(elem)
        script = self.get_sentencelist(strlist)
        return(script)
    
    
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
                pass
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


class NUCLEReader(object):
    def __init__(self):
        self.tagset = []
        self.CORPUS_PATH_RELATIVE = 'usr/share/NUCLE/shaped'
        self.DATAPATH = os.path.join(os.environ['HOME'], self.CORPUS_PATH_RELATIVE)
    


def main():
    import time
    total =time.time()
#    CLCReader().read_extract_all("AllOriginal", "IncorrectRV", "IncorrectVerbs", 
#                                 "AllCorrect")
    C = CLCReader()
    C.read()
    print(C.all_sents[0:10])
    endtime = time.time()
    print("\n\nOverall time %5.3f[sec.]"%(endtime - total))
    


if __name__ == "__main__":
    main()
