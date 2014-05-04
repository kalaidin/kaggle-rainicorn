#!/usr/bin/env python3
from scipy.optimize import fmin
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
import sys
from commons import *
import numpy as np
from sklearn.base import BaseEstimator, clone
from per_customer_eval import weighted_score


class LastQuoted(BaseEstimator):
    def __init__(self, feature_names=None):
        assert feature_names is not None, 'feature_names must be specified'
        self.feature_names = feature_names

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        assert (isinstance(X, csr_matrix))

        features = {fn: i for i, fn in enumerate(self.feature_names)}

        p = []
        for c in COVERAGE:
            c_indexi = [features['last_%s:%d' % (c, v)] for v in OPTIONS[c]]
            c_pred = np.array(X[:, c_indexi].dot(csr_matrix([OPTIONS[c]]).T).todense(), dtype='int')
            p.append(c_pred)
        return np.hstack(p)

    def __str__(self):
        return 'LastQuoted()'


class EachTaskIndependently(BaseEstimator):
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        task_estimators = []
        for i in range(y.shape[1]):
            e = clone(self.estimator)
            try:
                e.fit(X, y[:, i], sample_weight=sample_weight)
            except TypeError:
                print("Seems like %s doesn't support `sample_weight` argument. Ignoring it" % str(e),
                      file=sys.stderr)
                e.fit(X, y[:, i])
            print('train score %d: %0.3f' % (i, weighted_score(y[:, i], e.predict(X), sample_weight)))
            task_estimators.append(e)
        self.task_estimators = task_estimators

    def predict(self, X):
        return np.vstack([e.predict(X) for e in self.task_estimators]).T


def _to_ordinal(continuous_values, thresholds, labels):
    """ Convert continuous value into ordinal given its thresholds and labels
    >>> _to_ordinal(np.array([-1, 0, 0.6, 2]), np.array([0.5, 1]), np.array([0, 1, 2]))
    array([0, 0, 1, 2])"""
    assert thresholds.shape[0] + 1 == labels.shape[0]

    ordinal_values = np.zeros(continuous_values.shape, dtype='int') + labels[0]
    ordinal_values[continuous_values >= thresholds[-1]] = labels[-1]

    for i in range(thresholds.shape[0] - 1):
        ordinal_values[np.logical_and(continuous_values >= thresholds[i],
                                      continuous_values < thresholds[i + 1])] = labels[i + 1]
    return ordinal_values


def _best_threshold(actual_y, predicted_y, labels, sample_weight=None):
    """
    >>> c = lambda *x: np.array(x)
    >>> _best_threshold(c(0, 0, 1, 1), c(0.4, -1, 0.4001, 0.7), c(0, 1), None)
    array([0.4])
    """
    initial_thresholds = (labels[1:] + labels[:-1]) / 2
    f = lambda optim_thresh: weighted_score(actual_y, _to_ordinal(predicted_y, optim_thresh, labels),
                                            sample_weight)
    return fmin(f, initial_thresholds)

c = lambda *x: np.array(x)
_best_threshold(c(0, 0, 1, 1), c(0.4, -1, 0.4001, 0.7), c(0, 1), None)


class AsOrdinal(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1, "only one-dimensional target is supported"

        try:
            self.estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            print("%s doesn't support `sample_weight`. Ignoring it.")
            self.estimator.fit(X, y)

        y_hat = self.estimator.predict(X)

        self.labels = np.unique(y)
        self.labels.sort()

        self.thresholds = _best_threshold(y, y_hat, self.labels, sample_weight=sample_weight)

    def predict(self, X):
        _to_ordinal(self.estimator.predict(X), self.thresholds, self.labels)


if __name__ == '__main__':
    import doctest
    doctest.testmod()