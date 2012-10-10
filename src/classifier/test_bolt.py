#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/classifier/test_bolt.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

from nose.plugins.attrib import att
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from pprint import pformat
try: 
    import bolt
except:
    raise ImportError

class TestBoltClassifier:
    pass
