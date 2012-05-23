#! /usr/bin/python
# coding: utf-8
'''
Nyanco/lm_counter.py
Created on 11 Jan 2012

This module handles suspicious n-gram e.g. ("we", "VERB", "throughput")
and count frequency of candidates using command `ssgnc-search`


@author: Yu Sawai
@version: 0.001
'''
try:
    import subprocess
    import commands
    import os
    import cPickle as pickle
except ImportError:
    try:
        import pickle
    except ImportError:
        print("Failed to import pickle module from any place")
        


def ssgnc_search(querydic_i):
    '''
    Estimate counts of given queries on Web1T LM
    using ssgnc-search command through 'commands' module
    
    ARGS:
        querydict
    
    RETURNS:
        estimated_counts.txt
    '''
#    queries = ["I took the phone","I picked up the phone","I grasped the phone" "could join show", "could attend show", "could enter show", "and spareed money","and saved money", "ask to pay coffee","ask to buy coffee"]
    queries = []
    for k in querydic_i.keys():
        try:
            queries.append(str(querydic_i[k]))
        except UnicodeEncodeError:
            queries.append("_ _ _ _ _")
    print queries
    badqueries = []
    results = []
    errortup = ("ERROR","0")
    #    print queries
    with open("ssgnclog.log", "a") as f:
        for query in queries:
            output = commands.getoutput('echo "%s" | \
                                    ssgnc-search --ssgnc-order=FIXED \
                                    --ssgnc-max-num-results=25 \
                                    --ssgnc-min-freq=1 \
                                    /work/yu-s/cl/nldata/Web1T/index'%query)
            if output:
                try:
                    restuple = tuple(output.strip("\n").split("\t"))
                    if len(restuple) == 2:
                        results.append(restuple)
                    elif len(restuple) == 1:
                        results.append(errortup)
                    else:
                        results.append(errortup)
                except:
                    results.append(errortup)
#                print output
#                f.write(output)
#                results.append(output)
            else:
                results.append((query, "0"))
                badqueries.append(query)
#        f.write("-"*10+"\n")
#    with open("ssgnclog_failedqueries.log", "a") as debuglog:
#        for item in badqueries:
#            debuglog.write(item)
#            debuglog.write("\n")
#        debuglog.write("-"*10+"\n")
    return results

def ssgnc_result_read(estimated_count_file=None):
    '''
    Read info. of estimated counts from resultfile.
    
    ARGS:
        estimated_count_file
        
    RETURNS:
        dictionary of query and its counts
    '''
    pass

