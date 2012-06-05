# /usr/env/python
# coding: utf-8
'''
xml2raw.py

@author: yu-s

Convert English Gigaword data into raw text file
(a sentence per line)
'''
from lxml import etree
import os, glob
from StringIO import StringIO
from BeautifulSoup import BeautifulStoneSoup as Soup
import re

def addrootnode(file):
    with open(file, 'r') as f:
        _xmlstr = f.read()
    return '<ROOT>' + _xmlstr + '</ROOT>'


def readxml(xmlstr):
    _xmltree = etree.parse(StringIO(xmlstr))
    all_p = _xmltree.xpath('//P')
    return [p.text.strip('\n').replace('\n', ' ')+'\n' for p in all_p]


def readxmlsoup(xmlstr):
    soup = Soup(xmlstr)
    all_p = soup.findAll('p')
    return [p.text.strip('\n').replace('\n', ' ')+'\n' for p in all_p]



def globfiles(rootpath):
    filelist = glob.glob(rootpath+'*')
    return filelist


def output(filename, sentencelist):
    '''
    output(filename(as string afp_eng_xxxx),sentencelist)
    this will create a new raw file
    '''
    with open('raw/'+ filename + '_raw', 'w') as f:
        for line in sentencelist:
            f.write(line.encode('utf-8'))


def main():
    rootpath = ''
    filenames = [f for f in sorted(globfiles(rootpath)) ]
    print filenames
    for name in filenames:
        print 'Now processing %s'%name
        try:
            output(name, readxml(addrootnode(name)))
        except:
            print 'Using BeautifulStoneSoup instead of lxml.etree for: %s'%name
            output(name, readxmlsoup(addrootnode(name)))

if __name__=='__main__':
    main()
