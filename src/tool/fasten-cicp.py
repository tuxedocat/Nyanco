#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Converter of learners corpus
"""

__author__ = 'Yuta Hayashibe' 
__version__ = ""
__copyright__ = ""
__license__ = "GPL v3"


import collections
import sys
import time

class Line(list):
    SURFACE = 0
    POS = 3
    DEP_TO = 5
    REL = 6
    SEM = 9
    def __init__(self):
        self.dep = collections.defaultdict(list)
        pass

    def append(self, items):
        assert isinstance(items, list)

        dep_to = int(items[self.DEP_TO]) -1
        rel = items[self.REL]
        this_id = len(self)
        self.dep[dep_to].append(this_id)

        list.append(self, items)

    def rels(self, outputf):
#        print self.dep
        vchs = []
        time.sleep(0.01)
        for k, v in self.dep.items():
            to_id =  k
            from_ids = v
#            print "==="
#            print k
            if to_id < 0:
                continue
#                verb = "ROOT"
            else:
                if not self[to_id][self.POS].startswith("VB"):
                    continue

            _rels = {}
            for from_id in from_ids:
                arg = self[from_id][self.SURFACE]
                rel = self[from_id][self.REL]

                if rel == u"prep" :
                    for val in self.dep[from_id]:
                        if self[val][self.REL] == "pobj":
                            rel = arg
                            arg = self[val][self.SURFACE]
                            break
                elif rel == u"vch" :
                    vchs.append(from_id)
#                    for val in self.dep[from_id]:
#                        if self[val][self.REL] == "vch":
#                            rel = "vch"
#                            arg = (arg + " " + self[val][self.SURFACE])
#                            break

                _rels[rel] = arg

            if to_id in vchs:
                outputf.write("*\t")
            else:
                outputf.write("+\t")
            verb = self[to_id][self.SURFACE]
            verb_sem = self[to_id][self.SEM]
            outputf.write("%s\t%s" % (verb.lower(), verb_sem))
            for k, v in _rels.items():
                outputf.write("\t%s\t%s" % (k, v.lower()))
            outputf.write("\n")

SUFFIX = '.txt.parsed'
def cicp_extract(input_filename, output_filename, prefix):
    assert isinstance(input_filename, str)
    assert isinstance(output_filename, str)
    assert isinstance(prefix, str)

    outputf = open(output_filename, 'w')
    for line in open(input_filename):
        name = line.split()[0]

        "P/P08/P08-1069.txt.parsed"
        conference = name[0]
        year = name[1:3]

        fname = "%s/%s/%s%s/%s%s" % (prefix, conference, conference, year, name, SUFFIX)
        sent = Line()
        try:
            for line in open(fname, 'r'):
                if line == "\n":
#                print sent
                    sent.rels(outputf)
                    sent = Line()
                else:
                    sent.append(line[:-1].split()[1:])
        except:
            print "error at", fname



def extract(input_filenames, output_filename):
    if isinstance(input_filenames, str):
        input_filenames = [input_filenames]
    assert isinstance(input_filenames, list)
    assert isinstance(output_filename, str)

    outputf = open(output_filename, 'w')
    for fname in input_filenames:
        sent = Line()
        for line in open(fname, 'r'):
            if line == "\n":
#                print sent
                sent.rels(outputf)
                sent = Line()
            else:
                sent.append(line[:-1].split()[1:])



if __name__=='__main__':
    import sys
    argv = sys.argv
    argc = len(argv)

    USAGE = """Converter"""

    import optparse
    oparser = optparse.OptionParser(usage=USAGE)
    oparser.add_option('-i', dest = 'input_filename', action="append", default=[])
    oparser.add_option('-o', dest = 'output_filename')
    oparser.add_option('-p', dest = 'prefix')


    (opts, args) = oparser.parse_args()


    if len(opts.input_filename)==0:
        opts.input_filename = None
    elif len(opts.input_filename)==1:
        opts.input_filename = opts.input_filename[0]
    
    if (opts.input_filename and  opts.output_filename and opts.prefix):
        #extract(opts.input_filename, opts.output_filename)
        cicp_extract(opts.input_filename, opts.output_filename, opts.prefix)
    else:
        oparser.print_help()

    quit()