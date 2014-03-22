#!/usr/bin/env python3
from scipy.sparse import csr_matrix
from commons import *


class LastQuoted:
    def __init__(self, feature_names):
        self.features = {fn: i for i, fn in enumerate(feature_names)}

    def fit(self, X, y):
        pass

    def predict(self, X):
        p = []
        for c in COVERAGE:
            c_indexi = [self.features['last_%s:%d' % (c, v)] for v in OPTIONS[c]]
            c_pred = tuple(X[:, c_indexi].dot(csr_matrix([OPTIONS[c]]).T).todense())
            p.append(c_pred)
        return tuple(zip(*p))