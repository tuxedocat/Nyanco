# /usr/bin/env python
# coding: utf-8
'''
cleaner.py

This will clean out useless line of input file.
e.g.
* lines contain too much numbers
* match results
* tables of financial docs

'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, tuxedocat"
__credits__ = ["Yu Sawai"]
__license__ = "CC"
__version__ = "0.1"
__maintainer__ = "Yu Sawai"
__email__ = "yu-s@is.naist.jp"
__status__ = "Prototyping"


import logging
import re
import os
import sys
import time

re_table = re.compile(r'(\t{1,})')
re_numnum = re.compile(r'(\d{2,})')
re_score = re.compile(r'(\d{1,}-\d{1,})')
re_date = re.compile(r'(\d+/\d+/\d+)')

def notable(line):
    if not re_table.findall(line) : return True
    else: return False


def replace_num(line):
    if re_numnum.findall(line): return re_numnum.sub('numberstring', line)
    else: return line


def noscore(line):
    if not re_score.findall(line): return True
    else: return False


def replace_date(line):
    if re_date.findall(line): return re_date.sub('datestring', line)
    else: return line


def nospaces(line):
    if not '    ' in line and not '---' in line: return True
    else: return False


def noshortsent(line):
    if len(line) > 42: return True
    else: return False


def startswith_valid(line):
    if not line.startswith('-') and not line.startswith('GROUP'):return True
    else: return False


def is_all_ok(line):
    if notable(line) and noscore(line) and nospaces(line) and noshortsent(line) and startswith_valid(line): return True
    else: return False


def is_not_line_startswith_num(line):
    if not line.startswith('numberstring') and not line.startswith('numberstring', 5): return True
    else: return False


def filter(inputfile, outputfile):
    with open(inputfile, 'r') as infile:
        source = infile.readlines()
    outlist = []
    for i, line in enumerate(source):
        # print "processing document #%d"%(i)
        if is_all_ok(line):
            tmp = replace_num(replace_date(line))
            if tmp and is_not_line_startswith_num(tmp):
                outlist.append(tmp)
    with open(outputfile, 'w') as out:
        out.writelines(outlist)
    # with open(outputfile, 'w') as out:
        # with open(inputfile, 'r') as infile:
            # for i, line in enumerate(infile):
                # print "processing document #%d"%(i)
                # if is_all_ok(line):
                    # tmp = replace_num(replace_date(line))
                    # if tmp:
                        # out.write(tmp)


def main():
    usage = '''
    Usage:
    $ python cleaner.py inputfile outputfile
    '''
    st = time.time()
    try:
        input = sys.argv[1]
        output = sys.argv[2]
    except IndexError:
        print usage
        sys.exit()
    filter(input, output)
    et = time.time()
    print 'processing time %3.5f'%(et-st)


if __name__=='__main__':
    main()
