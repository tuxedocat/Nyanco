#! /usr/bin/env python
# coding: utf-8
'''
nyanco/tool/remove_xmltag_space.py

This removes spaces before/after xml tag, for regularization
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import os
import sys
import logging
import re
import fnmatch
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import cPickle as pickle


def main(inputdir):
    files = []
    for root, dirnames, filenames in os.walk(inputdir):
        for filename in fnmatch.filter(filenames, "*.xml"):
            files.append(os.path.join(root,filename))

    spaces_on_left = re.compile(r'(?<=\w)\s+<')
    spaces_on_right = re.compile(r'>\s+(?=\w)')

    for f in files:
        with open(f, 'r') as infile:
            txt = infile.read()
            txt = txt.decode("utf-8")
        try:
            # remove spaces on left side of the tag
            txt = spaces_on_left.sub('<', txt)
            # remove spaces on right side of the tag
            txt = spaces_on_right.sub('>', txt)
        except:
            logging.debug(("something wrong on ", f))
        with open(f, 'w') as outfile:
            outfile.write(txt)
            print "File %s is processed successfully!"%f
    logging.debug("processed %d files"%len(files))



if __name__=='__main__':
    argv = sys.argv
    argc = len(argv)
    USAGE = """
            python remove_xmltag_space.py -i /path/to/fce/root 

            CAUTION: This will OVERWRITE existing *.xml files
            """
    import optparse
    optp = optparse.OptionParser(usage=USAGE)
    optp.add_option('-i', '--input_prefix', dest = 'input_prefix')
    (opts, args) = optp.parse_args()
    if opts.input_prefix:
        main(opts.input_prefix)
        logging.debug("done")
    else:
        optp.print_help()
    quit()