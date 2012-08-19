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


class PasExtractor(object):
    """extract PAS from the parsed output of fanseparser"""
    def __init__(self, fname_in = None, fname_out = None):
        self.fname_in = fname_in
        self.fname_out = fname_out
        with open(self.fname_in, 'r') as f:
            self.raw = [line.split('\n') for line in f.read().split('\n\n')]


    def _extract(self, sent):
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
                    if tags[7] == "ROOT":
                        root = tags[1]
                        root_idx = int(tags[0])
                    elif 'ARG' in tags[12]:
                        args.append((tags[1], tags[12], int(tags[13])))
                except:
                    if sent == "":
                        pass
                    else:
                        logging.debug("Couldn't find the triple in sentence")
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

    def extract(self):
        """
        wrapper func. of extract 
        @takes
            self.raw :: a list of ['1\tWORD\t\t\t\t',...] (line for each sentence)
        """
        pasdiclist = [self._extract(sent) for sent in self.raw]
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
        pax = PasExtractor(os.path.join(input_prefix, f + ".txt.parsed"))
        tmpc = collections.Counter(pax.extract())
        pastriples_counter_native = pastriples_counter_native + tmpc
        output2file(input_prefix, output_prefix+"Native", pastriples_counter_native)

    for i, f in enumerate(foreign_list):
        logging.debug(('Foreign: Processing file no.\t %d (%d remaining...)'%(i+1,(num_ff-i-1))))
        pax = PasExtractor(os.path.join(input_prefix, f + ".txt.parsed"))
        tmpc = collections.Counter(pax.extract())
        pastriples_counter_foreign = pastriples_counter_foreign + tmpc
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
    print opts.mode
    if (opts.input_prefix and opts.output_prefix and opts.mode == "cicp"):
        cicp_extract(opts.input_prefix, opts.output_prefix)
    elif (opts.input_prefix and opts.output_prefix):
        extract(opts.input_prefix, opts.output_prefix)
    else:
        optp.print_help()
    quit()