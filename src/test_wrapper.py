#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/test_wrapper.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import logging
from pprint import pformat
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from wrapper import *
from nose.plugins.attrib import attr
import yaml

@attr("wrapper")
class TestExpWrapper:
    def setUp(self):
        path = "../sandbox/test.yaml"
        do_experiments(path)