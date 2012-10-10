#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/src/tool/test_onlineparser.py
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

import logging
from pprint import pformat
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import sys, os
import cPickle as pickle
import subprocess
from nose.plugins.attrib import attr
from online_fanseparser import OnlineFanseParser


class TestOnlineParser(object):
    def setUp(self):
        self.ofp = OnlineFanseParser(w_dir="../../sandbox/parse_online/")
        self.ofp.check_running()
        self.testtxt = [u'Dear Mr Robertson, I am writing to you first of all to thank you for the excellent programme that you have given us for the trip to London, especially for planning a visit to the science museum, which we believe is going to be very interesting.',
                        u'However, we have seen an advertisement for the London Fashion and Leisure Show and we would like to suggest including this event in the programme.', 
                        u'The show is going to be on Tuesday the 14 of March from 10.00 to 19.00 and we find it very interesting, because we will have the opportunity to see  , firstly about the latest fashions, secondly concerning leisure and sports wear and finally about make-up and hairstyles.', 
                        u'In addition to this, it is going to be a great opportunity for all of us, because for  students it is completely free.', 
                        u'We were thinking that maybe we can go early to visit the Science Museum and then go to the Central Exhibition Centre , where the event is going to take place!', 
                        u'We really look forward to receiving your answer as soon as possible!', 
                        u'Thank you very much !', 
                        u'Yours sincerely,']

    @attr("opsingle")
    def test_opsingle(self):
        pass


    @attr("opmulti")
    def test_opmulti(self):
        parseresult = []
        for txt in self.testtxt:
            parseresult.append(self.ofp.parse_one(txt))
        logging.debug(pformat(parseresult))
        self.ofp.clean()
        raise Exception

        
