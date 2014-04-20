#!/usr/bin/env python3
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
import sys

from per_customer_eval import *
from per_customer_models import *


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
            p = e.predict(X)
            yy = y[:, i]
            print('train score %d: %0.3f' % (i, weighted_score(yy, p, sample_weight)))
            task_estimators.append(e)
        self.task_estimators = task_estimators

    def predict(self, X):
        return np.vstack([e.predict(X) for e in self.task_estimators]).T


#eval_per_customer_model(LastQuoted)
eval_per_customer_model(EachTaskIndependently(SGDClassifier(n_iter=30)))
#eval_per_customer_model(EachTaskIndependently(SVC(kernel='linear')))
#eval_per_customer_model(EachTaskIndependently(LinearSVC(C=0.01)))


#OneVsRestClassifier()
