Nyanco
======

Verb selection error detector and suggestion maker


What's this?
----------

This set of scripts is to perform detection and suggestion for verb selection errors in English writings.

Quickstart
---------

To be written soon.

Libraries and packages required
-------------
* python 2.7 and above
* scikit-learn (dev)
* nltk
* pattern
* numpy/scipy
* bolt (optional)
* fanseparser
* senna

Method details
---------
Multiclass classifiers

* One Vs. the Rest classifier via. sklearn's linear models with Log-loss, Modified-huber-loss, and L1, and L2 regularization terms
