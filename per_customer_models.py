#!/usr/bin/env python3
from commons import *


class LastQuoted:
    def __init__(self, feature_names):
        self.features = {fn: i for i, fn in enumerate(feature_names)}

    def fit(self, X, y):
        pass

    def predict(self, X):
        p = []
        for i in range(X.shape[0]):
            last_quoted = []
            for c in COVERAGE:
                for v in OPTIONS[c]:
                    if X[i, self.features['last_%s:%d' % (c, v)]] == 1:
                        last_quoted.append(v)
                        break
            p.append(last_quoted)
        return p