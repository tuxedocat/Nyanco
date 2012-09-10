# /usr/bin/env python
# coding: utf-8
'''
cleaner.py

This will clean out useless line of input file.

from Bergsma et.al 2012, 
* first 3 lines with author's affiliation info. will be separated into *.afinfo
* remove footnotes, table, figure's caption
* remove lines with non-ASCII chars

And some features are added...
* remove lines without period (mostly they are contents of tables)
* remove lines starting with symbols (same as above) 
'''

__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, tuxedocat"
__maintainer__ = "Yu Sawai"
__email__ = "yu-s@is.naist.jp"
__status__ = "Prototype"


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import re
import os
import sys


re_table = re.compile(r'(\t{1,})')
re_numnum = re.compile(r'(\d{2,})')
re_score = re.compile(r'(\d{1,}-\d{1,})')
re_date = re.compile(r'(\d+/\d+/\d+)')
re_symbols = re.compile(r'(\s{1,3}:{1}\s{1,3})+')
re_Ref = re.compile(r'^(References)')
re_caption = re.compile(r'((Table)|(Figure)\s{1,3}\d{1,3}\s{1,3}:{1})+')
re_num = re.compile(r'\d{1}')
re_scattered = re.compile(r'([a-zA-Z]{1}\s{1}[a-zA-Z]{1}\s{1}){2,}') # e.g. :' M a c h i n e translation' will be matched
re_head = re.compile(r'^([A-Za-z0-9])')
re_tip = re.compile(r'\.$')
re_url = re.compile(r'(:\s?/\s?/)')


def notabs(line):
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


def not_tooshort(line):
    if len(line.split()) > 5 and len(line) > 30: return True
    else: return False


def starts_ends_with_valid(line):
    if re_head.match(line) and re_tip.search(line):return True
    else: return False


def is_not_scattered(line):
    '''
    remove lines with scattered words like "M a c h i n e translation"
    '''
    if not re_scattered.search(line): return True
    else: return False


def is_not_numbers(line):
    '''
    Detect line with too many numbers (assuming it would be a caption or content of table)
    '''
    if len(re_num.findall(line)) > 7: return False
    else: return True


def is_references(line):
    if re_Ref.match(line): return True
    else: return False


def is_not_caption(line):
    if not re_caption.match(line) and not re_url.search(line): return True
    else: return False


def is_not_line_startswith_num(line):
    if not line.startswith('numberstring') and not line.startswith('numberstring', 5): return True
    else: return False


def is_all_ok(line):
    if  notabs(line) and noscore(line) and nospaces(line) and not_tooshort(line) and starts_ends_with_valid(line) \
            and is_not_scattered(line) and is_not_caption(line) and is_not_numbers(line): return True
    else: return False


#------------------------------------------------------------------------------

def filter(inputfile, outfile_prefix):
    affiliation = []
    body_i = 10
    with open(inputfile, 'r') as infile:
        source = infile.readlines()
        for i, item in enumerate(source):
            if "Abstract" in item or i > 10:
                body_i = i + 1
                break
            else:
                affiliation.append(item)
    outlist = []
    for i, line in enumerate(source[body_i:]):
        try:
            if is_references(line):  # References is omitted
                break
            line.encode('ascii')     # lines with ASCII characters are passed
            if is_all_ok(line):
                outlist.append(line)
        except UnicodeError:
            pass
    with open(outfile_prefix+'.cleaned', 'w') as out:
        out.writelines(outlist)
    with open(outfile_prefix+'.affiliation','w') as af:
        af.writelines(affiliation)


def main():
    import time
    usage = '''
    Usage:
    $ python cleaner.py inputfile outfile_prefix
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
