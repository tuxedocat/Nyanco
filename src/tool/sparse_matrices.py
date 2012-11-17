#! /usr/bin/env python
# coding: utf-8
'''
sparse_matrices.py

Save and load scipy's sparse matrices
(Made this because somehow scikit-learn seems has no proper solution for save/loading its sparse matrices)
'''

import numpy as np
import scipy as sp


def save_sparse_matrix(filename, X):
    X_coo = X.tocoo()
    np.savez(filename, row=X_coo.row, col=X_coo.col, data=X_coo.data, shape=X_coo.shape)


def load_sparse_matrix(filename):
    m = np.load(filename)
    return sp.sparse.coo_matrix((m['data'], (m['row'], m['col'])), shape=m['shape'])
