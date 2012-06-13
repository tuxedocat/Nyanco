# coding: utf-8
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys,os,re,pickle
from gensim import corpora,models,similarities,utils