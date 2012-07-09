#!/usr/bin/env python
# coding: utf-8

'''
Nyanco/src/verbtuple_extractor.py

this script is for extracting verb, its ARG0 and ARG1 (3tuple)
from given parsing results by Fanseparser format

'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, tuxedocat"
__license__ = "GPL"
__email__ = "yu-s@is.naist.jp"
__status__ = "Prototype"


import os
import sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import cPickle as pickle

