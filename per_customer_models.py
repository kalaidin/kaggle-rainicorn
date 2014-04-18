#!/usr/bin/env python3
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from commons import *
import numpy as np


class LastQuoted:
    def __init__(self, feature_names=None):
        assert feature_names is not None, 'feature_names must be specified'
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

    def __str__(self):
        return 'LastQuoted()'


class SGD:
    def __init__(self, *params, feature_names=None, **kwd_params):
        assert feature_names is not None, 'feature_names must be specified'
        assert not params, "SGD doesn't take positional arguments. Use kwd args."

        self.features = {fn: i for i, fn in enumerate(feature_names)}
        self.models = {}
        self.params = params
        self.kwd_params = kwd_params

    def fit(self, X, y, sample_weight=None):
        assert isinstance(X, csr_matrix)
        assert isinstance(y, np.ndarray)
        if sample_weight is not None:
            assert isinstance(sample_weight, np.ndarray)

        for i, c in enumerate(COVERAGE):
            m = SGDClassifier(loss='log', **self.kwd_params)
            m.fit(X, y[:, i], sample_weight=sample_weight)
            self.models[c] = m

    def predict(self, X):
        assert isinstance(X, csr_matrix)

        p = []
        for c in COVERAGE:
            p.append(self.models[c].predict(X))

        return np.vstack(p).T

    def __str__(self):
        return 'SGDClassifier(%s)' % dict_tostring(self.kwd_params)