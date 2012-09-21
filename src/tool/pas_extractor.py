#! /usr/bin/env python
# coding: utf-8
'''
nyanco/tool/pas_extractor.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import os
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle
import collections
from collections import defaultdict
from pprint import pformat


class PasExtractor(object):
    """extract PAS from the parsed output of fanseparser"""
    def __init__(self, fname_in = None, fname_out = None):
        self.fname_in = fname_in
        self.fname_out = fname_out
        with open(self.fname_in, 'r') as f:
            self.raw = [line.split('\n') for line in f.read().split('\n\n')]
        self.col_surface = 1
        self.col_pos = 4
        self.col_depID = 6
        self.col_dep = 7
        self.col_srl = 12
        self.col_srldepID = 13
        self.col_ne = 10

    def _extract_simple(self, sent):
        """
        Extract PAS structure '(PREDICATE, ARGUMENT1, ARGUMENT2)'
        for each sentences.
        
        ROOT described in column 7 will be the PREDICATE, 
        ARGS are chosen by the ARGx tag on column 12 
        
        @takes
            sent :: a list of strings such as
                    ['1\tThe\t-\t-\tDT\t-\t...', '2\tcat\t...']

        """
        root = ""
        args = []           # in some case, there're multiple args
        root_idx = -100     # set default index for root, in case it doesn't exist
        tagtuples = [tuple(l.split('\t')) for l in sent] 
        if tagtuples:
            for tags in tagtuples:
                try:
                    if tags[self.col_dep] == "ROOT":
                        root = tags[1]
                        root_idx = int(tags[0])
                    elif 'ARG' in tags[self.col_srl]:
                        args.append((tags[1], tags[self.col_srl], int(tags[self.col_srldepID])))
                except:
                    pass
            # logging.info(('Arguments::',args)) 

            # loop for ARGS #
            # Look the index, if an ARG is for the ROOT (idx == ROOT_idx), 
            # add it as correct ARG
            pasdic = collections.defaultdict(tuple)
            pasdic['ROOT'] = root
            for arg in args:
                arg_idx = arg[2]
                arg_type = arg[1]
                arg_surface = arg[0]
                if arg_idx == root_idx:
                    pasdic[arg_type] = (arg_surface)
            return pasdic
        else:
            pass


    def __format_preddic(self, preddict):
        pasdic_list = []
        moc = ("","","","")
        for pkey in preddict:
            out = {}
            out["PRED"] = (pkey[self.col_surface], pkey[self.col_pos], pkey[self.col_dep], pkey[self.col_ne])
            try:
                a0 = preddict[pkey]["ARG0"]
                out["ARG0"] = (a0[self.col_surface], a0[self.col_pos], a0[self.col_dep], a0[self.col_ne])
            except KeyError:
                out["ARG0"] = moc 
            try:
                a1 = preddict[pkey]["ARG1"]
                out["ARG1"] = (a1[self.col_surface], a1[self.col_pos], a1[self.col_dep], a1[self.col_ne])
            except KeyError:
                out["ARG1"] = moc 

            pasdic_list.append(out)
        return pasdic_list

    def _extract_full(self, sent, verb=""):
        """
        Extract PAS structure like Liu et.al 2010

        'I have opened an American bank account in Boston.'
            A0_I_opened
            opened_A1_account
            opened_AM-LOC_in

        Firstly, filter the tags which contain ARG0, and ARG1 respectively.
        Then, from self.col_srldepID, find each predicates
        The output format will be like...
        [a list of dicts {"PAS name":(<surface>, <POS>, <dep_tag>, <NE tag>)} ]
                    column in files:          1      4        7           10

                pasdic_list = [{"PRED":("have", "VBP","conj","have.03"), "ARG0":("we", "PRP", "nsubj", "_"), "ARG1":("right", "NN", "dobj", "_")},
                        {"PRED":("sum", "VB", "ROOT", "sum.01"), "ARG0":None, "ARG1":("are", "VBP", "ccomp", "be.01")},
                        {"PRED":("lead", "VB", "infmod", "lead.01"), "ARG0":None, "ARG1":("life", "NN", "dobj", "_") },
                        {"PRED":("break", "VB", "infmod", "break.02"), "ARG0":None, "ARG1":("into", "IN", "prep", "_")} ]



        """
        tagtuples = [tuple(l.split('\t')) for l in sent] 
        self.tmp_tt = tagtuples
        self.tmp_ARG0 = []
        self.tmp_ARG1 = []
        self.tmp_PRED = defaultdict(dict)
        self.relationslist = []
        self.pasdic_list = [] # each list element is a dictionary
        if tagtuples:
            for tt in tagtuples:
                try:
                    if tt[self.col_srl] == "ARG0":
                        self.tmp_ARG0.append(tt)
                        predidx = int(tt[self.col_srldepID]) - 1
                        self.tmp_PRED[tagtuples[predidx]].update({"ARG0":tt})
                    elif tt[self.col_srl] == "ARG1":
                        self.tmp_ARG1.append(tt)
                        predidx = int(tt[self.col_srldepID]) - 1
                        self.tmp_PRED[tagtuples[predidx]].update({"ARG1":tt})
                except Exception as e:
                    logging.debug(pformat(e.string))
        self.pasdic_list = self.__format_preddic(self.tmp_PRED)
        return self.pasdic_list


    def extract(self):
        """
        wrapper func. of extract 
        @returns
            self.paslist :: a list of tuples (ROOT, ARG0, ARG1)
        """
        pasdiclist = [self._extract_simple(sent) for sent in self.raw]
        self.paslist = [(pdic['ROOT'], pdic['ARG0'], pdic['ARG1']) for pdic in pasdiclist
                        if pdic['ROOT'] and pdic['ARG0'] and pdic['ARG1'] ]
        return self.paslist


def output2file(input_prefix, output_prefix, counter_obj):
    import cPickle as pickle
    # Store the Counter object into cPickle binary
    with open(os.path.join(input_prefix, output_prefix + 'PAS.pickle'), 'wb') as pkl:
        pickle.dump(counter_obj, pkl)

    # Output as tab-separated plain text file
    # format is...
    # PRED_SURFACE\tARG0_SURFACE\tARG1_SURFACE\tCOUNT
    # Will be sorted by count
    with open(os.path.join(input_prefix, output_prefix + 'PAS.tsv'), 'w') as tsv:
        for k, v in sorted(counter_obj.iteritems(), key=lambda x: x[1], reverse=True):
            outstr = '\t'.join(k) + '\t' + str(v) + '\n'
            tsv.write(outstr)




class PEmod(PasExtractor):
    """
    Another version of PasExtractor, for a sentence
    """
    def __init__(self, fname="", verb=""):
        if fname:
            self.fname_in = fname
            self.fname_out = ""
        with open(self.fname_in, "r") as f:
            self.raw = [line for line in f.read().split("\n") if line]
        self.col_surface = 1
        self.col_pos = 4
        self.col_depID = 6
        self.col_dep = 7
        self.col_srl = 12
        self.col_srldepID = 13
        self.col_ne = 10
        self.verb = verb


    def extract(self):
        """
        wrapper func. of _extract_simple
        @returns
            self.paslist :: a list of tuples (ROOT, ARG0, ARG1)
        """
        pasdiclist = [self._extract_simple(self.raw)] 
        if pasdiclist:
            self.paslist = [(pdic['ROOT'], pdic['ARG0'], pdic['ARG1']) for pdic in pasdiclist
                            if pdic and (pdic['ROOT'] and pdic['ARG0'] and pdic['ARG1']) ]
            return self.paslist
        else:
            pass


    def extract_full(self):
        """
        wrapper func. of _extract_full
        @returns
            self.paslist :: a list of dictionary 
        """
        return self._extract_full(self.raw)


class OnlinePasExtractor(PasExtractor):
    """
    Another version of PasExtractor, for a sentence, coop with OnlineFanseParser

    * mapping (full to simple)
        * Surface: 1, 1
        * POS: 4, 2
        * DEP_TO: 6, 4
        * DEPENDENCY: 7, 3
        * NE: 10, 5
        * SRL: 12, 6
        * SRL_REL: 13, 7
    """
    def __init__(self, taglist):
        self.raw = taglist
        self.col_surface = 1
        self.col_pos = 2
        self.col_dep = 3
        self.col_depID = 4
        self.col_srl = 6
        self.col_srldepID = 7
        self.col_ne = 5


    def extract_full(self):
        """
        wrapper func. of _extract_full
        @returns
            self.paslist :: a list of dictionary 
        """
        return self._extract_full(self.raw)








def extract(input_prefix, output_prefix):
    """
    Wrapper function of whole process.

    @takes
        input_prefix :: string of path-to-files, e.g. '../../cl/EnglishGigaword/raw/afp_eng'
        (files are assumed to have '.parsed' suffix)

        output_prefix :: string of prefix of the output file name, e.g. 'afp_eng_' 
        ('PAS.tsv' will be added automatically)
        (the output will be saved as tab-separated text and python pickle file)
    """
    import glob
    files = glob.glob(os.path.join(input_prefix,'*.parsed'))
    num_f = len(files)
    pastriples_counter = collections.Counter()
    for i, f in enumerate(files):
        logging.debug(('Processing file no. %d (%d remaining...)'%(i+1,(num_f-i-1))))
        pax = PasExtractor(f)
        tmpc = collections.Counter(pax.extract())
        pastriples_counter = pastriples_counter + tmpc
    output2file(input_prefix, output_prefix, pastriples_counter)


def cicp_extract(input_prefix, output_prefix):
    native_list = open(os.path.join(input_prefix, 'Native.txt'), 'r').readlines()
    native_list = [fn.strip('\n')[2:] for fn in native_list]
    foreign_list = open(os.path.join(input_prefix, 'Foreign.txt'), 'r').readlines()
    foreign_list = [fn.strip('\n')[2:] for fn in foreign_list]
    num_nf = len(native_list)
    num_ff = len(foreign_list)

    pastriples_counter_native = collections.Counter()
    pastriples_counter_foreign = collections.Counter()
    for i, f in enumerate(native_list):
        logging.debug(('Native:  Processing file no.\t %d (%d remaining...)'%(i+1,(num_nf-i-1))))
        try:
            pax = PasExtractor(os.path.join(input_prefix, f + ".txt.parsed"))
            tmpc = collections.Counter(pax.extract())
            pastriples_counter_native = pastriples_counter_native + tmpc
        except IOError:
            logging.debug(('Native: "No such file exists" at file  %s '%(f)))
            pass
    output2file(input_prefix, output_prefix+"Native", pastriples_counter_native)

    for i, f in enumerate(foreign_list):
        logging.debug(('Foreign: Processing file no.\t %d (%d remaining...)'%(i+1,(num_ff-i-1))))
        try:
            pax = PasExtractor(os.path.join(input_prefix, f + ".txt.parsed"))
            tmpc = collections.Counter(pax.extract())
            pastriples_counter_foreign = pastriples_counter_foreign + tmpc
        except IOError:
            logging.debug(('Foreign: "No such file exists" at file  %s '%(f)))
    output2file(input_prefix, output_prefix+"Foreign", pastriples_counter_foreign)



if __name__=='__main__':
    import sys
    argv = sys.argv
    argc = len(argv)
    # logging.disable('debug')
    USAGE = """
            python pas_extractor.py -i ../../cl/EnglishGigaword/raw/afp_eng -o afp_eng_

            -i --input_prefix :: string of path-to-files
            (files are assumed to have '.parsed' suffix)

            -o --output_prefix :: string of prefix of the output file name, e.g. 'afp_eng_' 
            ('PAS.tsv' will be added automatically)
            (the output will be saved as tab-separated text and python pickle file)

            -m --mode :: if this is used for cicp
            """

    import optparse
    optp = optparse.OptionParser(usage=USAGE)
    optp.add_option('-i', '--input_prefix', dest = 'input_prefix')
    optp.add_option('-o', '--output_prefix', dest = 'output_prefix')
    optp.add_option('-m', '--mode', dest = 'mode')

    (opts, args) = optp.parse_args()
    if len(opts.input_prefix)==0:
        opts.input_prefix = None
    elif len(opts.input_prefix)==1:
        opts.input_prefix = opts.input_prefix[0]
    # print opts.mode
    if (opts.input_prefix and opts.output_prefix and opts.mode == "cicp"):
        cicp_extract(opts.input_prefix, opts.output_prefix)
    elif (opts.input_prefix and opts.output_prefix):
        extract(opts.input_prefix, opts.output_prefix)
    else:
        optp.print_help()
    quit()