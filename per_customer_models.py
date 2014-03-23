#!/usr/bin/env python3
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelBinarizer
from commons import *
import numpy as np


class LastQuoted:
    def __init__(self, feature_names):
        self.features = {fn: i for i, fn in enumerate(feature_names)}

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        assert (isinstance(X, csr_matrix))

        p = []
        for c in COVERAGE:
            c_indexi = [self.features['last_%s:%d' % (c, v)] for v in OPTIONS[c]]
            c_pred = np.array(X[:, c_indexi].dot(csr_matrix([OPTIONS[c]]).T).todense(), dtype='int')
            p.append(c_pred)
        return np.hstack(p)


class SGD:
    def __init__(self, feature_names):
        self.features = {fn: i for i, fn in enumerate(feature_names)}
        self.models = {}

    def fit(self, X, y, sample_weight=None):
        assert isinstance(X, csr_matrix)
        assert isinstance(y, np.ndarray)
        if sample_weight is not None:
            assert isinstance(sample_weight, np.ndarray)

        for i, c in enumerate(COVERAGE):
            m = SGDClassifier(loss='log', n_iter=30)
            m.fit(X, y[:, i], sample_weight=sample_weight)
            self.models[c] = m

    def predict(self, X):
        assert isinstance(X, csr_matrix)

        p = []
        for c in COVERAGE:
            p.append(self.models[c].predict(X))

        return np.vstack(p).T