#!/usr/bin/env python
# coding:utf-8
__author__ = 'yu-s'
__version__ = '0.001'
__description__ = '''
Count freq. on LM
'''
from collections import defaultdict
import random 
from irstlm import getSentenceScore 
import datetime

class Webfrequency(object):
    '''
    this class is for estimating frequency of given list of queries 
    on the Web1T corpus using IRSTLM
    '''
    def __init__(self, irstlm_LM, dict_queries, dict_id2words):
        '''
        Constructor args:
            irstlm_LM: object created by initLM
            dict_queries: 
            dict_id2words: {0:'give',1:'offer',....}

        '''
        self.lm = irstlm_LM 
        self.dict_queries = dict_queries
        self.resultdict = defaultdict(list)
        self.id2words = dict_id2words


    def countall(self):
        '''Return score on LM for queries
        each query is given as a list: [w-2, w-1, cand, w1, w2]
        ARGS:
            Nothing
        RETURNS:
            scores on LM for each queries
        '''
        for wid in self.dict_queries:
            qlist = self.dict_queries[wid][:]
            for ql in qlist:
                correct = self.id2words[wid]
                eachresult = {}
                eachresult.update({'correct':correct, 'rawtext': " ".join(ql)})
                for candid in self.id2words:
                    cand = self.id2words[candid] 
                    q_cand = ql[:]
                    try:
                        # c_index = q_cand.index(cand)
                        c_index = 2
                        q_cand.pop(c_index)
                        q_cand.insert(c_index, cand)
                    except:
                        print q_cand
                        pass
                    query4LM  = " ".join(q_cand)
                    count = getSentenceScore(self.lm, query4LM)
                    eachresult.update({cand:count})
                self.resultdict[wid].append(eachresult)
        print self.resultdict.items()

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


from sklearn import metrics
def make_report_ngram(resultdict,id2word):
    '''
    Calculate accuracy over the test set, make report of the test
    Write log files of failed-cases, and the report

    ARGS:
        resultdict: a dictionary obtained from the object Webfrequency()
        id2word:    a dictionary

    '''
    report = defaultdict() 
    _result = []
    _outputlabels = []
    _truelabels = []
    word2id = {}
    _accuracy_list = []
    # Obtain result for each test instances
    for wid in resultdict:
        _result.append([check_outputs(eachdict) for eachdict in resultdict[wid]])
        _outputlabels += [check_outputs(eachdict)[2] for eachdict in resultdict[wid]]
        _truelabels += [ed['correct'] for ed in resultdict[wid] ]
    # Calc. accuracy and write failed queries into a file
    for wid, word in id2word.iteritems():
        word2id[word] = wid
        _result4word = set(_result[wid])
        _corrects = [t[0] for t in _result4word if t[0] is True]
        _fails = [t[1] for t in _result4word if t[0] is False]
        _accuracy = float(len(_corrects)) / float(len(_result4word))
        _accuracy_list.append(_accuracy)
        report[word]={'accuracy': _accuracy, 'failedcases':_fails}
    for word in report:
        print 'WORD %s'%word
        print "accuracy %3.4f"%(report[word]["accuracy"])
        with open('train/ngram_fails_%s.log'%word,'w') as rf:
            for line in report[word]["failedcases"]:
                rf.write(line+"\n\n")
    # Convert labels from string to integer IDs
    _outputlabels = [word2id[word] for word in _outputlabels]
    _truelabels = [word2id[word] for word in _truelabels]
    # Write the classification_report using sklearn.metrics
    d = datetime.datetime.today()
    exp_time = d.strftime("%Y%m%d_%H%M")
    with open('train/ngram_'+exp_time+'.log','w') as f:
        for word in report:
            f.write("WORD: \t"+word+"\n")
            f.write("Accuracy: \t %3.4f\n" %(report[word]["accuracy"]))
        f.write('\n'+80*"="+'\n\n')
        _names = [id2word[wid] for wid in id2word]
        f.write(metrics.classification_report(_truelabels, _outputlabels, target_names=_names))
        f.write('\n\nAccuracy over the test set: %3.4f' % (sum(_accuracy_list)/len(_accuracy_list)))
    return report


def check_outputs(eachdict):
    '''
    args:
        eachdict: {'correct':'<label>', '<cand1>': <score:float>,...}
    return:
        True if correct label is #1 candidate
        (True or False, query, output label)
    '''
    _lm_outputs = []
    for k, v in eachdict.iteritems():
        if not (k == 'correct' or k == 'rawtext'):
            _lm_outputs.append((k,v))
    _result = [t for t in sorted(_lm_outputs, key=lambda x:x[1], reverse=True)]
    _correctlabel = eachdict['correct']
    if _result[0][0] == _correctlabel:
        return (True, eachdict['rawtext'], _result[0][0])
    else:
        return (False, eachdict['rawtext'], _result[0][0])

