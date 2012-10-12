#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/tool/online_fanseparser.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import logging
from pprint import pformat
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import sys, os
import cPickle as pickle
import subprocess
from copy import deepcopy
from nose.plugins.attrib import attr



class OnlineFanseParser(object):
    def __init__(self, w_dir=""):
        try:
            FProot = os.environ["FANSEPARSER"]
        except KeyError:
            print "set FANSEPARSER path first!!"
            FProot = ""
        self.MAX_MEMORY = "-Xmx7000m"
        self.PORT_NUMBER = "5776"
        self.CLASSPATH = FProot + "fanseparser-0.2.2_mod.jar"
        self.CLIENTCLASSPATH = FProot + "fanseparser-0.2.2_mod.jar"
        self.WORDNET = FProot + "data/wordnet3/"
        self.SENTENCE_READER = "tratz.parse.io.TokenizingSentenceReader"
        self.SENTENCE_WRITER = "tratz.parse.io.DefaultSentenceWriter"
        self.POS_MODEL = FProot + "posTaggingModel.gz"
        self.PARSE_MODEL = FProot + "parseModel.gz"
        self.POSSESSIVES_MODEL = FProot + "possessivesModel.gz"
        self.NOUN_COMPOUND_MODEL = FProot + "nnModel.gz"
        self.SRL_ARGS_MODELS = FProot + "srlArgsWrapper.gz"
        self.SRL_PREDICATE_MODELS = FProot + "srlPredWrapper.gz"
        self.PREPOSITION_MODELS = FProot + "psdModels.gz"
        self.servercmd = ["java", self.MAX_MEMORY, "-cp", self.CLASSPATH, "tratz.parse.SimpleParseServer",
                          "-port", self.PORT_NUMBER, "-wndir", self.WORDNET, "-posmodel", self.POS_MODEL,
                          "-parsemodel", self.PARSE_MODEL, "-possmodel", self.POSSESSIVES_MODEL,
                          "-nnmodel", self.NOUN_COMPOUND_MODEL, "-psdmodel", self.PREPOSITION_MODELS,
                          "-srlargsmodel", self.SRL_ARGS_MODELS, "-srlpredmodel", self.SRL_PREDICATE_MODELS]
        self.clientcmd = ["java", self.MAX_MEMORY, "-cp", self.CLIENTCLASSPATH, "tratz.parse.SimpleParseClient", self.PORT_NUMBER] # Input txt will be added to the last
        self.tmpdir = os.path.abspath(w_dir)
        self.tmpname = "parsertmp.txt"
        self.tmpfilename = os.path.abspath(os.path.join(self.tmpdir, self.tmpname))

    def check_running(self):
        self.fanseserver_str = "tratz.parse.SimpleParseServer"
        ps_out = subprocess.check_output(["ps", "-A"])
        if not self.fanseserver_str in ps_out:
            print "Starting fanseparser server.... on port %s"%self.PORT_NUMBER
            scmd = deepcopy(self.servercmd)
            # logging.debug(pformat(scmd))
            self.server = subprocess.Popen(scmd, stderr=subprocess.PIPE)
            while True:
                line = self.server.stderr.readline()
                if line != "":
                    if line == "Server started, waiting to accept connections...\n":
                        break
                else:
                    break
        else:
            print "Fanseparser server is running...."

    def _parse(self, filename=""):
        try:
            parsecmd = deepcopy(self.clientcmd)
            parsecmd.append(filename)
            subprocess.call(parsecmd)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        finally:
            pass


    def parse_one(self, string=""):
        parsed = []
        if string:
            with open(self.tmpfilename, "w+") as wf:
                wf.write(string)
            self._parse(filename=self.tmpfilename)
            with open(self.tmpfilename+".parsed", "r") as parsed_f:
                parsed = [line.strip("\n") for line in parsed_f.readlines() if line != '\n']
        return parsed


    def clean(self):
        self.server.terminate()
        logging.debug("Terminated fanseparser server....")




def parse_online(w_dir=""):
    if not w_dir:
        print "Need working directory..."
    else:
        wdpath = os.path.join(w_dir,"parsertmp")
        if not os.path.exists(wdpath):
            os.makedirs(wdpath)
