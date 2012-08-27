#! /usr/bin/env python
# coding: utf-8
'''
Nyanco/preprocessor.py
Created on 10 Jan 2011

This module handles a list of lists:
    [[script1], [script2],...]
        [script n] = ["This cat", ("is", "was", "TV"), "born yesterday."]
@author: Yu Sawai
@version: 0.001
'''

try:
    import os
    import subprocess
    import re
    import nltk
    from nltk.corpus import verbnet as vn
except ImportError:
    print("You'd better find valid libraries")


def preprocess4ssgnc(scripts):
    '''
    A wrapper function which literally do everything...
    '''
    filtered_scr = specify_errors(scripts)
    errorindex = finderror(scripts)
    qdict = make_baseqdict(concat_context(filtered_scr, errorindex))
    queries_i_2_2, queries_c_2_2 = query_formulation(qdict)
    return queries_i_2_2, queries_c_2_2
    
def query_formulation(qdict):
    '''
    this is for formulating queries with various criteria
    
    ARGS:
        qlist: base query list
    RETURNS:
        queries_i_m_n: query for incorrect verb with m pred. context and n succ.
                        context
        queries_c_m_n: query for correct verb
        (those queries are given as dictionary)
    '''
    def find_v(tuple):
        vindex = [x for x,y in enumerate(tuple) if hasattr(y, "lower")]
        return vindex[0]
    
    def concat(tuple, n, vindex=1):
        try:
            pred = reduce(lambda x,y:" "+x+" "+y, tuple[vindex-1][-n:])
            succ = reduce(lambda x,y:" "+x+" "+y, tuple[vindex+1][:n])
            if pred.endswith(""):
                return (pred+" "+tuple[vindex]+succ)
            else:
                return (pred+tuple[vindex]+succ)
        except TypeError:
            # when context around the verb is too short, just ignore it.
            pass
        
#    def getsynsets(verb):
#        try:
#            synset = vn.lemmas(vn.classids("%s"%verb))
#        except IndexError:
#            synset = []
#            print "verb %s doesn't have synonyms."
#        except:
#            synset = []
#            print "something is wrong"
#        return synset
        
    queries_i_2_2 = dict( (k,concat(tup[0], 2, find_v(tup[0]))) 
                          for k, tup in qdict.items())
    queries_c_2_2 = dict( (k,concat(tup[1], 2, find_v(tup[1]))) 
                          for k, tup in qdict.items())
    return queries_i_2_2, queries_c_2_2
    
    
def make_baseqdict(querycontexts):
    '''
    ARGS:
        querycontext: a list consisted of tuples 
            ("context before a Verb", (<i>,<c>,"RV"), "context after a Verb")
    RETURNS:
        basequerydict: a dict of tuples 
        keys = ("incorrect", "correct", "type") :
        values = ( (['w_pred1',...], 'V_incorrect', ['w_succ1',...]),
                   (['w_pred1',...], 'V_correct', ['w_succ1',...]) )
    '''
    def correct_v(qtuple): return qtuple[1][1]
    
    def incorrect_v(qtuple): return qtuple[1][0]

    def getkey(index, qtuple): return tuple([qtuple[1][0],qtuple[1][1]]+[index])
    
    def predcontext(qtuple): return [ str for str in qtuple[0].split(" ")
                                    if not (str == "" or str == " ")]
    
    def succcontext(qtuple): return [ str for str in qtuple[2].split(" ")
                                    if not (str == "" or str == " ")]

    def baseq_i(qtuple): return tuple([predcontext(qtuple)[-6:], 
                                       incorrect_v(qtuple), 
                                       succcontext(qtuple)[:6]])
    
    def baseq_c(qtuple): return tuple([predcontext(qtuple)[-6:], 
                                       correct_v(qtuple), 
                                       succcontext(qtuple)[:6]])
    
    keylist = [getkey(i, qtuple) for i, qtuple 
               in enumerate(querycontexts) 
               if len(predcontext(qtuple))>=2 and len(succcontext(qtuple))>=2]
    baseqlist = [(baseq_i(qtuple),baseq_c(qtuple)) for qtuple 
                 in querycontexts 
                 if len(predcontext(qtuple))>=2 and len(succcontext(qtuple))>=2]
    basequerydict = dict((k,v) for k,v in zip(keylist,baseqlist))
#    print basequerydict
    return basequerydict

    
def specify_errors(scripts):
    '''
    an utility function to select RV errors only...
    other errors are temporaly just corrected using correction information.
    
    ARGS:
        scripts: a list containes lists of exam scripts consisted of 
        strings and tuples
    RETURNS:
        filtered: lowercased, filtered scripts
    '''
    
    filtered = []
    for script in scripts:
        eachscript = []
        for elem in script:
            try:
                eachscript.append(elem.lower())
            except AttributeError:
                if elem[2] == "RV":
                    eachscript.append((x.lower() for x in elem))
                elif elem[1] is not None:
                    eachscript.append(elem[1].lower())
        filtered.append(eachscript)
    return filtered


def finderror(scripts):
    '''
    ARGS: 
        scripts: list of script-lists
    RETURNS: 
        errorindex: list of lists of tuples of the index and tuple of error element
    '''
    errorindex = [ [(i, t) for i, t in enumerate(script) 
                   if (not hasattr(t, "lower") and t[2] == "RV" 
                       and not t[0] == "" and not t[1] == "" 
                       and not t[0] == None) ] 
                  for script in scripts]
    return errorindex


def concat_context(scripts, errorindex):
    '''
    RETURNS: tuples of contatenated contexts for each problematic words
    '''
    contextlist = []
    for scriptindex, errorlist in enumerate(errorindex):
        for errortuple in errorlist:
            try:
                pred = scripts[scriptindex][errortuple[0]-1]
                succ = scripts[scriptindex][errortuple[0]+1]
                etuple = errortuple[1]
                contextlist.append((pred.lower(),
                                   etuple,
                                   succ.lower().strip(".").strip(",") ))
                print pred, errortuple, succ
            except AttributeError:
                contextlist.append(("",
                                    ("","",""),
                                    ""))
#                print "pred. or succ. isn't string type"
            except IndexError:
                contextlist.append(("",
                                    ("","",""),
                                    ""))
                print "index error"
    return contextlist