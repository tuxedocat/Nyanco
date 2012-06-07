# /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
import os


re_table = re.compile(r'(\t{1,})')
re_numnum = re.compile(r'(\d{2,})')
re_score = re.compile(r'(\d{1,}-\d{1,})')
re_date = re.compile(r'(\d+/\d+/\d+)')

def notable(line):
    if not re_table.findall(line): return True
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
    if not '    ' in line: return True
    else: return False


def noshortsent(line):
    if len(line) > 30: return True
    else: return False


def is_all_ok(line):
    if notable(line) and noscore(line) and nospaces(line) and noshortsent(line): return True
    else: return False

def filter():
    with open('afp_eng_2010_raw_clean.txt', 'w') as out:
        with open('afp_eng_2010_raw_cat', 'r') as infile:
            for i, line in enumerate(infile):
                print "processing document #%d"%(i)
                if is_all_ok(line):
                    tmp = replace_num(replace_date(line))
                    if tmp:
                        out.write(tmp)


