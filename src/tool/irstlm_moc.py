#coding: utf-8
'''
Nyanco/tool/irstlm_moc.py


This module is a fake, just for deceiving the detector when executing non-linux environment
where the IRSTLM python binding can not work properly
'''
__author__ = "Yu Sawai"
__copyright__ = "Copyright 2012, Yu Sawai"
__version__ = "0.1"
__status__ = "Prototyping"

from pprint import pformat
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
from nose.plugins.attrib import attr
import random 


initmssg_fake = """blmt\n
loadbin()\n
lmtable::loadbin_dict()\n
dict->size(): FAKE_IRSTLM_BLM\n
loadbin_level (level 1)\n
loading SOME_FAKE 1-grams\n
done (level1)\n
loadbin_level (level 2)\n
loading SOME_FAKE 2-grams\n
done (level2)\n
loadbin_level (level 3)\n
loading SOME_FAKE 3-grams\n
done (level3)\n
loadbin_level (level 4)\n
loading SOME_FAKE 4-grams\n
done (level4)\n
loadbin_level (level 5)\n
loading SOME_FAKE 5-grams\n
done (level5)\n
done\n
OOV code is xxxxxx\n"""

deletemssg_fake =   """Deleting FAKE_LM....\n
Deleting FAKE_LM has been completed\n
                    """

def initLM(*args):
    # logging.debug(pformat(initmssg_fake))
    print initmssg_fake

def deleteLM(*args):
    # logging.debug(pformat(deletemssg_fake))
    print deletemssg_fake

def getSentenceScore(*args):
    return random.uniform(-30.0, -5.0)

def getFileScore(*args):
    return random.uniform(-30.0, -5.0)


class TestIrstlmMoc:
    def __init__(self):
        pass

    @attr("irstlm_moc")
    def test_score(self):
        initLM()
        self.query = ["the cat is a black",
                      "the dog is a black",
                      "the man is a black"]
        for q in self.query:
            print getSentenceScore(q)
        deleteLM()
        raise Exception