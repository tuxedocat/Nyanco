#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/corpusreader2.py
Created on 29 Aug 2012

LOG:
    Sep 5: preprocessor has been moved to preprocessor2.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "2.1"
__status__ = "Prototyping"


import os
import cPickle as pickle
import json
from pprint import pformat
from datetime import datetime
logfilename = datetime.now().strftime("corpusreader2_log_%Y%m%d_%H%M.log")
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG,
                    filename='../log/'+logfilename)
try:
    from lxml import etree
    print("running with lxml.etree")
except ImportError:
    print("Import error")
    quit()


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


def store_to_pickle(path="", obj=None):
    '''
    store obtained various grained sentence lists using pickles.
    '''
    if obj:
        fname = path + datetime.now().strftime("_%Y%m%d_%H%M") + ".pickle"
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)
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
        self.skip_recursive = True


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


    def _extract(self, et):
        """
        returns ElementTree of <NS> if given etree element of <p> has NS element,
        and checks if the given <p> tree has recursive tree, just reject it.
        """
        plist = et.xpath("./text()|./NS") # this gives ["text", <NS>, "text"...] sometimes <NS> contains recursion
        paragraph = []
        for e in plist:
            if isinstance(e, str):
                paragraph.append(unicode(e))
            elif isinstance(e, etree._Element):
                check = e.xpath(".//NS")
                if check:  # Element which has recursive correction.... reject entire paragraph
                    if self.skip_recursive:
                        paragraph = None
                        return paragraph 
                    else:
                        paragraph.append(e)
                else:
                    paragraph.append(e)
        return paragraph


    def extract_script(self, certain_etree):
        '''
        A method to obtain an element tree of exam-script of learner itself.
        returns listed script: ["some text", <NS element>, ...]
        '''
        try:
            script = []
            answers = certain_etree.xpath("//coded_answer")
            for ans in answers:
                paragraphs = ans.findall("p")
                answer = []
                for p in paragraphs:
                    tmp = self._extract(p)
                    if tmp == None:
                        logging.debug(pformat("recursive annotation is detected, rejected an answer script"))
                        answer = []
                        break
                        # logging.debug(pformat("recursive annotation is detected, added marker"))
                        # tmp = "{RECURSIVE_ANNOTATION}"
                        # answer.append(tmp)
                    else:
                        answer += tmp
                script += answer
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
        if self.inputstring:
            return(self.inputstring)
        else:
            return("")


class IfElem(object):
    '''
    This is an utility class for xmlreader.
    For etree.Element object, method 'corrpair' will obtain 
    a correction pair: (org, corr, $errortype)
        (when etree_element is <NS ***><i>foo</i><c>bar</c></NS>)
    '''
    
    def __init__(self, etree_element):
        self.elem = etree_element
        # print pformat(etree.tostring(self.elem))
        # print pformat(etree_element.values())

    def check_errortype(self, etelem):
        try:
            errortype = unicode(etelem.values()[0])
            if errortype:
                return(errortype)
            else:
                return(u"")
        except :
            # logging.debug(pformat("Couldn't get error type", etree.tounicode(etelem)))
            # print pformat(etree.tounicode(etelem))
            return(errortype)


    def corrpair(self):
        original = u""
        correction = u""
        try:
            errortype = self.check_errortype(self.elem)
        except:
            errortype = u""
        try:
            self.elem.text
        except AttributeError:
            raise AttributeError
        try:
            if self.elem.find("i").text:
                original = unicode(self.elem.find("i").text)
            else:
                original = u""
        except AttributeError:
            original = u""
        try:
            correction = unicode()
            if self.elem.find("c").text:
                correction = unicode(self.elem.find("c").text)
            else:
                correction = u""
        except AttributeError:
            correction = u""
        correction_pair = (original,correction, errortype)
        if errortype == u"RV":
            print pformat(correction_pair)
            if original=="" and correction=="" and self.elem.text != None:
                correction_pair = (unicode(self.elem.text), unicode(self.elem.text), errortype, "M")
        return(correction_pair)

# ===================================================================================================

def read(corpus_dir="", output_dir="", working_dir=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    C = CLCReader(corpus_dir=corpus_dir, output_dir=output_dir, working_dir=working_dir)
    C.read()
    store_to_pickle(path=output_dir+"/all_sents", obj=C.all_sents)
    store_to_pickle(path=output_dir+"/index", obj=C.listindexdict)
    return C.all_sents, C.listindexdict


def main(corpus_dir="", output_dir="", working_dir="", preprocess=False):
    import preprocessor2
    corpus, filedict = read(corpus_dir=corpus_dir, output_dir=output_dir, working_dir=working_dir)
    if preprocess:
        corpus_as_dict = preprocessor2.preprocess(corpus, filedict.values(), modelist=["Incorrect_RV", "Gold"])
        store_to_pickle(output_dir+"/corpusdict_", corpus_as_dict)


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
    ap.add_argument("-i", '--input_dir', action="store",
                    help="string of path to FCE corpus dir")
    ap.add_argument("-o", '--output_dir', action="store",
                    help="string of path-to-dir")
    ap.add_argument("-d", '--working_dir', action="store",
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