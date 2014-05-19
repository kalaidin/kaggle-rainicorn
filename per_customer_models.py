#!/usr/bin/env python3
from itertools import chain, combinations, product
from numpy.core.umath import logical_or
from scipy.sparse import csr_matrix
import sys
from commons import *
import numpy as np
from sklearn.base import BaseEstimator, clone


class LastQuotedOnCategoricalLastX(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y, sample_weight=None):
        labels = np.unique(y)
        labels.sort()
        self.columns = {}
        for lab in labels:
            cat_y = y == lab
            best_matches = 0
            self.columns[lab] = -1
            for c_id in range(X.shape[1]):
                x = X[:, c_id]
                if sorted(np.unique(x)) == [0, 1]:
                    matches = ((x == 1) == cat_y).sum()
                    if matches > best_matches:
                        self.columns[lab] = c_id
                        best_matches = matches
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for label in self.columns:
            y[X[:, self.columns[label]] == 1] = label
        return y



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


#def _best_threshold(actual_y, predicted_y, labels, sample_weight=None):
#    """
#    >>> c = lambda *x: np.array(x)
#    >>> _best_threshold(c(0, 0, 1, 1), c(0.4, -1, 0.4001, 0.7), c(0, 1), None)
#    array([0.4])
#    """
#    initial_thresholds = (labels[1:] + labels[:-1]) / 2
#    f = lambda optim_thresh: weighted_score(actual_y, _to_ordinal(predicted_y, optim_thresh, labels),
#                                            sample_weight)
#    return fmin(f, initial_thresholds, disp=0)


#class AsOrdinal2(BaseEstimator):
#    """estimator must be a regressor not a classifier"""
#    def __init__(self, estimator):
#        self.estimator = estimator
#
#    def fit(self, X, y, sample_weight=None):
#        assert isinstance(y, np.ndarray)
#        assert len(y.shape) == 1, "only one-dimensional target is supported"
#
#        try:
#            self.estimator.fit(X, y, sample_weight=sample_weight)
#        except TypeError:
#            print("%s doesn't support `sample_weight`. Ignoring it.")
#            self.estimator.fit(X, y)
#
#        y_hat = self.estimator.predict(X)
#
#        self.labels = np.unique(y)
#        self.labels.sort()
#
#        self.thresholds = _best_threshold(y, y_hat, self.labels, sample_weight=sample_weight)
#
#        print('thresholds', self.thresholds)
#
#    def predict(self, X):
#        return _to_ordinal(self.estimator.predict(X), self.thresholds, self.labels)


class AsOrdinal(BaseEstimator):
    """estimator must be a classifier with implemented predict_proba() method"""
    def __init__(self, estimator, use_all_data=True):
        self.estimator = estimator
        self.estimators = {}
        self.use_all_data = use_all_data

    def iterate_labels(self):
        yield from zip(self.labels[:-1], self.labels[1:])

    def fit(self, X, y, sample_weight=None):
        assert isinstance(y, np.ndarray)
        assert len(y.shape) == 1, "only one-dimensional target is supported"

        self.labels = np.unique(y)
        self.labels.sort()

        for l0, l1 in self.iterate_labels():
            if self.use_all_data:
                l0_vs_v1 = np.array(y >= l1, dtype='int')
                new_x = X
            else:
                l0_vs_v1 = np.zeros(y.shape) - 1.0
                l0_vs_v1[y == l0] = 0
                l0_vs_v1[y == l1] = 1
                ii = logical_or(y == l0, y == l1)
                new_x = X[ii]


            e = clone(self.estimator)
            try:
                e.fit(new_x, l0_vs_v1, sample_weight=sample_weight)
            except TypeError:
                print("%s doesn't support `sample_weight`. Ignoring it.")
                e.fit(new_x, l0_vs_v1)
            self.estimators[(l0, l1)] = e

    def predict_proba(self, X):
        comparative_probs = np.zeros((X.shape[0], len(self.labels) - 1), dtype='float')
        for i, (l0, l1) in enumerate(self.iterate_labels()):
            comparative_probs[:, i] = self.estimators[(l0, l1)].predict_proba(X)[:, 1]

        p = np.zeros((X.shape[0], len(self.labels)), dtype='float')
        p[:, 0] = 1.0
        for i, (l0, l1) in enumerate(self.iterate_labels()):
            proba_l1 = self.estimators[(l0, l1)].predict_proba(X)[:, 1]
            proba_l0 = 1.0 - proba_l1
            k = proba_l1 / proba_l0
            p[:, i + 1] = p[:, i] * k

        return p / p.sum(axis=1)[:, np.newaxis]  # normalise row-wise

    def predict(self, X):
        label_indexi = self.predict_proba(X).argmax(axis=1)
        return np.array(self.labels)[label_indexi]


def create_interactions(coverage_name='last_X', order=2, from_coverages=COVERAGE):
    """
    >>> create_interactions(coverage_name='_X', from_coverages=['A', 'B', 'C'])
    [('_A:0', '_B:0'), ('_A:0', '_B:1'), ('_A:1', '_B:0'), ('_A:1', '_B:1'), ('_A:2', '_B:0'), ('_A:2', '_B:1'), ('_A:0', '_C:1'), ('_A:0', '_C:2'), ('_A:0', '_C:3'), ('_A:0', '_C:4'), ('_A:1', '_C:1'), ('_A:1', '_C:2'), ('_A:1', '_C:3'), ('_A:1', '_C:4'), ('_A:2', '_C:1'), ('_A:2', '_C:2'), ('_A:2', '_C:3'), ('_A:2', '_C:4'), ('_B:0', '_C:1'), ('_B:0', '_C:2'), ('_B:0', '_C:3'), ('_B:0', '_C:4'), ('_B:1', '_C:1'), ('_B:1', '_C:2'), ('_B:1', '_C:3'), ('_B:1', '_C:4')]"""
    combs_with_values = [list(product(*[[coverage_name.replace('X', '%s:%d' % (c, i)) for i in OPTIONS[c]]
                                        for c in comb]))
                         for comb in combinations(from_coverages, order)]
    return list(chain(*combs_with_values))


class JointEstimator(BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators
    def fit(self, X, y, sample_weight=None):
        assert len(self.estimators) == y.shape[1], 'There must be an estimator for each y task'
        self.fitted_estimators = []
        for i, e in enumerate(self.estimators):
            estimator = clone(e)
            try:
                estimator.fit(X, y[:, i], sample_weight=sample_weight)
            except:
                print('%s no sample_weight. Use without it.' % str(estimator))
                estimator.fit(X, y[:, i])
            self.fitted_estimators.append(estimator)
    def predict(self, X):
        return np.hstack([e.predict(X)[:, np.newaxis] for e in self.fitted_estimators])


def compress(columns):
    """Converts many columns into a single one
    >>> compress(np.array([[0, 1], [2, 3]]))
    array([10, 32])"""
    n = columns.shape[1]
    single_column = np.zeros(columns.shape[0], dtype='int')
    for i in range(n):
        single_column += columns[:, i] * 10**i
    return single_column


def decompress(values):
    """Reverse of compress
    >>> decompress(np.array([10, 32]))
    array([[0, 1],
           [2, 3]])
    >>> r = np.random.random_integers(0, 5, size=(10, 3))
    >>> np.all(r == decompress(compress(r)))
    True"""
    v = np.array(values)
    columns = []
    while np.max(v) > 0:
        columns.append(np.mod(v, 10)[:, np.newaxis])
        v //= 10
    return np.hstack(columns)


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def create_wrapper_and_unwrapper(*rulesets):
    """returns 2 functions.
    >>> w, uw = create_wrapper_and_unwrapper([2, 0], [1])
    >>> w(np.array([[0, 1, 4], [2, 3, 0]]))
    array([[ 4,  1],
           [20,  3]])
    >>> uw(w(np.array([[0, 1, 4], [2, 3, 0]])))
    array([[0, 1, 4],
           [2, 3, 0]])"""
    if not rulesets:
        return identity, identity
    def wrapper(y):
        return np.hstack([compress(y[:, cs])[:, np.newaxis] for cs in rulesets])
    def unwrapper(y):
        return np.hstack([decompress(y[:, i]) for i in range(y.shape[1])])[:, inv(list(chain(*rulesets)))]
    return wrapper, unwrapper


if __name__ == '__main__':
    import doctest
    doctest.testmod()