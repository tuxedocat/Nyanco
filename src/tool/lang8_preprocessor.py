#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/tool/lang8_preprocessor.py

Dirty preprocessing for lang8 data, done in iPython interactive shell mainly....
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from datetime import datetime
import logging
import os
import difflib
from pprint import pformat
from collections import defaultdict
import cPickle as pickle
import shelve # note: use only DbfinameShelve since there's no bsddb in pine servers
import re

# re_sline = re.compile(r"(\[sline\].*\[/sline\])", re.UNICODE)
# re_tags = re.compile(r"(\[(|\/)\w+-\w+\])", re.UNICODE)
re_l8tags = re.compile(r"(\[(|\/)\w+-\w+\])|(\[sline\].*\[/sline\])", re.UNICODE)

def removetags(l8str):
    return re_l8tags.sub("", l8str.replace("\n", "", 10), 100)

def filterout():
    try:
        db = shelve.DbfilenameShelf("../../sandbox/lang8/eng.shelve")
        for idx, k in enumerate(db.iterkeys()):
            try:
                print "Processing #%d sentence"%(idx+1)
                db[k]["correct_notags"] = removetags(db[k]["correct_raw"])
            except Exception, e:
                print e
                pass
    except Exception, e:
        print e
    finally:
        db.close()


# Get diff from original/correction pair sentences
def _decode_diff(_o, _a, dt):
    o_i1 = dt[1]; o_i2 = dt[2]
    a_i1 = dt[3]; a_i2 = dt[4]
    return _o[o_i1:o_i2], _a[a_i1:a_i2]


def _tdiff(_o, _a):
    replaced = [t for t in difflib.SequenceMatcher(None, _o, _a).get_opcodes() if t[0] == 'replace']
    if replaced:
        tmp = [_decode_diff(_o, _a, dt) for dt in replaced]
        # print tmp
        for tt in tmp:
            o = tt[0]
            a = tt[1]
            _res = [t for t in zip(o,a) if "VB" in t[0][1] and "VB" in t[1][1]]
            if _res:
                print _res
                for r in _res:
                    return r
            # print  [(t[0][0], t[1][0]) for t in tmp if "VB" in t[0][1] and "VB" in t[1][1]]


def get_diff(_o, _a):
    if hasattr(_o, "upper"):
        _o = _o.split()
    if hasattr(_a, "upper"):
        _a = _a.split()
    isfound = False
    while isfound is False:
        _tdiff(_o, _a)




# Used in iPython
def savetoshelve(docids, orgs, annots, shelvedb):
    for idx, docid in enumerate(docids):
        did = str(docid)
        for sentid, sentpair in enumerate(zip(orgs[idx], annots[idx])):
            if sentpair[1]:
                sid = str(sentid)
                try:
                    shelvedb[did+"_"+sid] = {"original":unicode(sentpair[0]),
                                             "correct_raw":unicode("".join(sentpair[1])),
                                             "correct":"",
                                             "comments":"",
                                             "flags":[],
                                             "proficiency_level":"",
                                             "L1":""}

                    print did+"_"+sid
                except Exception, e:
                    print e

                    
# import shelve
# import sqlite3
# import sqlalchemy
# from elixir import *

# metadata.bind = "sqlite:///eng.sqlite"
# metadata.echo = True

# class Sentence(Entity):
#     """
#     A simple representation for sentences in the lang-8 corpus
#     """
#     doc_id = Field(Integer)
#     sent_id = Field(Integer)
#     original_text = Field(UnicodeText)
#     annotated_raw = Field(UnicodeText)    
#     annotated_verb = Field(UnicodeText)
#     comments = Field(UnicodeText)
#     flags = Field(Unicode(30))
#     proficiency_level = Field(Unicode(10))
    
#     def __repr__(self):
#         return '<Sentence::\ndoc_id "%d" \tsent_id "%d" \norg_text "%s" \nannotaion "%s"\nflags "%s">'%(self.doc_id, self.sent_id, self.original_text, self.annotated_raw, self.flags)



# """
# session.close()
# for docid, tuple in enumerate(zip(docidlist, orgsentlist, annotatedlist)):
#     for sentid, sentpair in enumerate(zip(tuple[1], tuple[2])):
#         try:
#             if sentpair[1]:
#                 Sentence(doc_id=tuple[0], 
#                          sent_id=sentid, 
#                          original_text=unicode(sentpair[0]), 
#                          annotated_raw=unicode("".join(sentpair[1])), 
#                          annotated_verb=u"", 
#                          comments=u"", 
#                          flags=u"", 
#                          proficiency_level=u"")
#         except:
#             pass
# session.commit()
# """
