Nyanco
======

Verb selection error detector and suggestion maker


What's this?
----------
This set of scripts is to perform detection and suggestion for verb selection errors in English writings.

Quickstart
---------

### Perform quick test on preconfigured test set
In ipython shell,

``` py
import suggest_each
result = suggest_each.suggest_for_testset(corpuspath="../sandbox/kj_vlxc_corpus.pkl2", 
                                          cspath="../sandbox/classify/ConfusionSets_Lang8_FceVoc500Only_r50.pkl2", 
                                          modelrootpath="../sandbox/classify/models_l8r50_5gram_DA/", 
                                          modeltype="sgd_modifiedhuber_l2", 
                                          features=["5gram", "chunk"])
```

#### Some notes

* This requires "SENNAPATH" env. value to be specified.
* Copora are not included in this repo.

Libraries and packages required
-------------
* python 2.7.x
* scikit-learn (0.13 and higher, or dev-channel)
* nltk 2.0.1
* Pattern (Web mining module)
    * [Get Pattern](http://www.clips.ua.ac.be/pattern)
* numpy/scipy
* SENNA Parser
    * [Get SENNA](http://ml.nec-labs.com/senna/)
<!-- * fanseparser  -->


Method details
---------
Multiclass classifiers

* One Vs. the Rest classifier via. sklearn's linear models with Log-loss, Modified-huber-loss, and L1, and L2 regularization terms
