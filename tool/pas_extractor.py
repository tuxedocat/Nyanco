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
                    logging.debug("error")
            logging.debug(('Arguments::',args)) 
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
            return None


    def extract(self):
        """
        wrapper func. of extract 
        @takes
            self.raw :: a list of ['1\tWORD\t\t\t\t',...] (line for each sentence)
        """
        pasdiclist = [self._extract(sent) for sent in self.raw]
        self.paslist = [(pdic['ROOT'], pdic['ARG0'], pdic['ARG1']) for pdic in pasdiclist]
        print self.paslist
        return self.paslist


def extract(input_dir, output_prefix):
    import glob
    files = glob.glob(input_dir)
    pastriples_counter = collections.Counter()
    for f in files:
        pax = PasExtractor(f)
        pastriples_counter += pax.extract()
    return pastriples_counter


#===============================================================================

class TestPasExtractor:
    def setUp(self):
        import os, sys
        import glob
        import collections
        relpath = '../sandbox/pas/testdat*'
        self.testfile = glob.glob(relpath)
        self.eg1 = '../sandbox/pas/afp_eng_201012_raw.parsed'

    def test_extract1(self):
        pax = PasExtractor(self.testfile[0])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('loved', 'We', 'cat')])
        assert triples == expected

    def test_extract2(self):
        pax = PasExtractor(self.testfile[1])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('loved', 'We', 'cat'),('loved', 'We', 'cat'),('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat'),('loved', 'We', 'cat'),('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'dog')])
        assert triples == expected

    def test_extract3(self):
        '''
        test for complicated structure (which will be ignored)
        '''
        pax = PasExtractor(self.testfile[2])
        result = pax.extract()
        triples = collections.Counter(result)
        expected = collections.Counter([('urged', 'Royce', 'House'), ('insisted', 'Burns', 'doing')])
        assert triples == expected

    def test_extract456(self):
        '''
        test for multiple files
        '''
        triples = collections.Counter()
        for f in self.testfile[3:7]:
            pax = PasExtractor(f)
            result = pax.extract()
            tmpc = collections.Counter(result)
            triples = triples + tmpc
        expected = collections.Counter([('urged', 'Royce', 'House'), ('insisted', 'Burns', 'doing'), ('loved', 'We', 'cat'),
                                        ('loved', 'We', 'cat'),('loved', 'We', 'cat')])
        assert triples == expected

    def test_extract_gigaword(self):
        pax = PasExtractor(self.eg1)
        result = pax.extract()
        triples = collections.Counter(result)
        print triples.most_common(1)
        assert False

if __name__=='__main__':
    import sys
    argv = sys.argv
    argc = len(argv)

    USAGE = """
                $python pas_extractor.py -i input_filename -o output_filename
            """

    import optparse
    optp = optparse.OptionParser(usage=USAGE)
    optp.add_option('-i', dest = 'input_filename', action = "append", default = [])
    optp.add_option('-o', dest = 'output_filename')
    optp.add_option('-p', dest = 'prefix')

    (opts, args) = optp.parse_args()
    if len(opts.input_filename)==0:
        opts.input_filename = None
    elif len(opts.input_filename)==1:
        opts.input_filename = opts.input_filename[0]
    
    if (opts.input_filename and  opts.output_filename and opts.prefix):
        #extract(opts.input_filename, opts.output_filename)
        cicp_extract(opts.input_filename, opts.output_filename, opts.prefix)
    else:
        optp.print_help()
    quit()