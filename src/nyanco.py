#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/nyanco.py
Created on 11 Jan 2012


@author: Yu Sawai
@version: 0.001
'''
try:
    import subprocess
    import os
    import cPickle as pickle
    # nyanco modules
    import corpusreader
    import preprocessor
    import lm_counter
    import postprocessor

except ImportError:
    print("Failed to import pickle module from any place")


def main():
    corpusrdr = corpusreader.CLCReader()
    corpusrdr.read()
    queries_i_2_2, queries_c_2_2 = preprocessor.preprocess4ssgnc(corpusrdr.all_sents)
#    print queries_i_2_2
#    print queries_c_2_2
    print len(queries_c_2_2.keys())
    result_of_i_2_2 = lm_counter.ssgnc_search(queries_i_2_2)
    result_of_c_2_2 = lm_counter.ssgnc_search(queries_c_2_2)
#    print zip(result_of_i_2_2,result_of_c_2_2)
#    resultbase = [q for q in result_of_i_2_2 if not (q[0] == "0") ]
    num_all = len(result_of_i_2_2)
    num_nonzero = len([q for q in result_of_i_2_2 if not (q[1] == "0") ])
    goodlist = [t for t in zip(result_of_i_2_2,result_of_c_2_2)
                if (int(t[1][1]) > int(t[0][1])) and (t[0][1] != "0") and (t[1][1] != "0") ]
    num_correct = len(goodlist)
    print "\n"*3
    print "Processed RV tagged correction pairs: ", num_all
    print "Num. of non-zero result: ", num_nonzero
    print "Num. of gold prevails original: ", num_correct
    print "Acc:: n(Freq_i < Freq_c)/n(RV_tagged): ", 1.0*num_correct/num_all
    print "Non-Zero-Correction-rate:: n(correct)/n(nonzero): ", 1.0*num_correct/num_nonzero
    return True


if __name__=="__main__":
    import time
    start_time = time.time()
    main()
    print "done in %6.3f[sec.]"%(time.time()-start_time)