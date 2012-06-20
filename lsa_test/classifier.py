#! /usr/env/bin python
# coding: utf-8
import os, sys
from gensim import corpora, models, utils
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import pickle as pkl
import glob


class LSAconverter(object):
    '''
    This class handles raw text data of training and test data
    Then convert them using given LSA model 
    
    Usage:
        >>>LSAconverter(modelpath, modeldictpath, trainingdata_path, testdata_path)
    FIXME:
        Currently gensim.models.Lsimodel.load() doesn't work correctly so Pickle will be used
        But it is inefficient
    '''
    def __init__(self, modelpath, dictpath, traindir_path, testdir_path):
        self.tr_dir = traindir_path
        self.test_dir = testdir_path
        self.modelpath = modelpath
        self.dictpath = dictpath
        # self.LSAmodel = models.LsiModel.load(modelpath)
        self.LSAmodel = pkl.load(open(modelpath,'rb'))
        self.modeldict = corpora.Dictionary.load(dictpath)


    def readRawdata(self):
        traindatalist = glob.glob(os.path.join(self.tr_dir,'*.txt'))
        self.labeldict = {id: os.path.split(f)[-1].strip('.txt') for id, f in enumerate(traindatalist)}
        self.tr_files =  {id: f for id, f in enumerate(traindatalist)}
        self.tr_docs = {id: [l.strip('\n') for l in open(f,'r').readlines()] for id, f in enumerate(traindatalist)}


    def _line_preprocess(self, string, blankword):
        return utils.simple_preprocess(string.strip('\n').replace(' '+blankword+' ', ' ')) 


    def preprocess(self):
        '''
        create BoW vector of given train and test text data
        assuming one line in the input file as one instance
        '''
        self.tr_docs_bow = {}
        for k in self.tr_docs:
            blankword = self.labeldict[k].lower()
            self.tr_docs_bow[k] = [self.modeldict.doc2bow(self._line_preprocess(l, blankword)) for l in self.tr_docs[k]]


    def convert2LSAVector(self):
        '''
        Convert BoW vectors into LSA vector
        '''
        self.tr_lsvectors = {}
        for k in self.tr_docs_bow:
            self.tr_lsvectors[k] = self.LSAmodel[self.tr_docs_bow[k]]
        

    def save2SVMlight(self):
        '''
        Save each lsa vectors to SVMlight format files
        '''
        for k in self.tr_lsvectors:
            name = self.labeldict[k]
            path = os.path.join(self.tr_dir,name+'.svmlight')
            corpora.SvmLightCorpus.serialize(path, self.tr_lsvectors[k]) 
        pkl.dump(self.labeldict, open(os.path.join(self.tr_dir,'labeldict.pkl'), 'wb'))



class SvmLightCorpusH(object):
    def __init__(self, datadir):
        self.labeldict = pkl.load(open(os.path.join(datadir,'labeldict.pkl'), 'rb'))
        self.datadir = datadir
    
    def label(self):
        self.svmlightdata = {}
        for id, name in self.labeldict.iteritems():
            self.svmlightdata[id] = open(os.path.join(self.datadir, name+'.svmlight'), 'r').readlines()
            with open(os.path.join(self.datadir, name+'.labelled'), 'w') as out:
                for line in self.svmlightdata[id]:
                    if line.find('ERROR') == -1:
                        output = line.replace('0', str(id), 1)
                        out.write(output)
                    else:
                        pass



# =============================================================================
# =============================================================================
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn import metrics, svm, multiclass
import random
import glob
from collections import defaultdict
import numpy
import time
class Classifier(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.labeldict = pkl.load(open(os.path.join(filepath,'labeldict.pkl'), 'rb'))
        self.results = defaultdict(list) 
        self.instances = {}
        self.texts = {}
        self.report = {}

    def readinstances(self):
        for key, value in self.labeldict.iteritems():
            _x , _y = load_svmlight_file(os.path.join(self.filepath, value+'.labelled'))
            self.instances[value] = {'x': _x.toarray(), 'y': _y}
            with open(os.path.join(self.filepath, value + '.txt'),'r') as rf:
                _text = rf.readlines()
                self.texts[value] = _text


    def classify(self, NUM_TRAIN=1000):
        self.NUM_TRAIN = NUM_TRAIN
        _xtrain = []
        _ytrain = []
        _xtest = []
        _ytest = []
        self._testtexts = []
        for k, xypair in self.instances.iteritems():
            x, y = xypair['x'], xypair['y']
            _xtrain.append(x[:NUM_TRAIN])
            _ytrain.append(y[:NUM_TRAIN])
            _xtest.append(x[NUM_TRAIN:NUM_TRAIN+200])
            _ytest.append(y[NUM_TRAIN:NUM_TRAIN+200])
            self._testtexts += self.texts[k][NUM_TRAIN:NUM_TRAIN+200]
        xtrain = numpy.concatenate(_xtrain)
        ytrain = numpy.concatenate(_ytrain)
        xtest = numpy.concatenate(_xtest)
        ytest = numpy.concatenate(_ytest)
        self.output = multiclass.OneVsRestClassifier(svm.LinearSVC()).fit(xtrain,ytrain).predict(xtest)
        self.testlabel = ytest
        _labels = [self.labeldict[k] for k in self.labeldict]
        self.report['table'] = metrics.classification_report(self.testlabel, self.output, target_names=_labels)
        print self.report['table'] 
        self.accuracy = float(len([t for t in zip(self.testlabel, self.output) if t[0] == t[1] ])) / len(self.output)
        print self. accuracy


    def failedcases(self):
        print self._testtexts
        self.report['fails'] = []
        for elemnum, (true, pred) in enumerate(zip(self.testlabel, self.output)):
            if not pred == true:
                try:
                    print self._testtexts[elemnum]
                    self.report['fails'].append(self._testtexts[elemnum]) 
                except:
                    print 'no _testtexts element'


def classificationtest(path):
    import datetime
    CLS = Classifier(path)
    CLS.readinstances()
    CLS.classify(NUM_TRAIN=5000)
    d = datetime.datetime.today() 
    exp_time = d.strftime("%Y%m%d_%H%M")
    with open(os.path.join(path, 'classification_'+exp_time+'.log'), 'w') as f:
        f.write(CLS.report['table'])
        f.write('\n\n')
        f.write('Accuracy over the instances:\t %3.4f'%CLS.accuracy)
    CLS.failedcases()
    with open(os.path.join(path, 'errors_'+exp_time+'.log'), 'w') as f:
        for elem in CLS.report['fails']:
            f.write(elem + '\n')
 



def main():
    '''
    usage:

    $ python classifier.py Path2Modelfile Path2Modeldictionary Path2traindir Path2testdir
    '''
    MODELPATH = sys.argv[1]
    MODELDICPATH = sys.argv[2]
    TRAINDIR = sys.argv[3]
    TESTDIR = sys.argv[4]
    SVMDDIR = TRAINDIR
    LSC = LSAconverter(MODELPATH, MODELDICPATH, TRAINDIR, TESTDIR)
    LSC.readRawdata()
    LSC.preprocess()
    LSC.convert2LSAVector()
    LSC.save2SVMlight()
    SVMLH = SvmLightCorpusH(SVMDDIR)
    SVMLH.label()


if __name__=='__main__':
    main()

