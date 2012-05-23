'''
Nyanco/xmltest.py
Created on 31 Dec 2011

@note: this is being abandoned
@author: tuxedocat
'''
from corpusreader import CLCReader
import unittest
import commands
import pprint
import os
import re
import sys
import lxml



class Test(unittest.TestCase):

    def setUp(self):
        pass
    
    def traverse_test(self):
        CLCReader().read_extract_all("AllCorrect")
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()