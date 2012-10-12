#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/tool/altgen.py
Created on 12 Oct 2012


Retrieve alternative words for given word using wordnet and verbnet
via nltk

'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0"
__status__ = "Prototyping"

from pprint import pformat
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from nose.plugins.attrib import attr
import random


class AlternativeGenerator(object):
    pass