#!/usr/bin/env python3
from contextlib import contextmanager
import os
from random import random
from time import time
import pandas as pd
from sklearn.cross_validation import KFold
import numpy as np

CV_GROUPS_COUNT = 3

CUSTOMER_ID = 'customer_ID'

DATA_DIR = 'data'
CV_DIR = 'cv'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
DELIM = ','

TERMINATION_PROB = 0.4

COVERAGE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
OPTIONS = {
    'A': (0, 1, 2),
    'B': (0, 1),
    'C': (1, 2, 3, 4),
    'D': (1, 2, 3),
    'E': (0, 1),
    'F': (0, 1, 2, 3),
    'G': (1, 2, 3, 4)

}


def j(*paths):
    """Shortcut for os.path.join"""
    return os.path.join(*paths)


def read_data(file_name=TRAIN_FILE):
    """file argument is from {'train', 'test'}"""
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path, index_col=False, sep=DELIM)


def convert_to_test(data):
    """By throwing out last transactions"""
    i = 0
    test_indices = []
    for customer_ID, g in data.groupby(CUSTOMER_ID):
        test_indices.append(i)  # first transaction is always in dataset
        for k in range(1, len(g) - 1):
            if random() < TERMINATION_PROB:
                break
            test_indices.append(i + k)
        i += len(g)
    return data.iloc[test_indices]


def cv(data):
    assert isinstance(data, pd.core.frame.DataFrame)
    customers = tuple(set(data[CUSTOMER_ID].values))
    for train_i, test_i in KFold(len(customers), shuffle=True, n_folds=CV_GROUPS_COUNT):
        train_customers = {customers[i] for i in train_i}
        test_customers = {customers[i] for i in test_i}
        train = data[data[CUSTOMER_ID].isin(train_customers)]
        test = convert_to_test(data[data[CUSTOMER_ID].isin(test_customers)])
        yield train, test


def not_implemented_raiser(message=''):
    def f():
        raise NameError(message=message)
    return f


@contextmanager
def timer(process_name):
    t = time()
    yield
    print('"%s" finished in %0.1f seconds' % (process_name, time() - t))


def equal_rows(mat1, mat2):
    return np.abs(mat1 - mat2).sum(axis=1) == 0


def dict_tostring(d, sep=', '):
    """
    >>> dict_tostring({1: 'b', 'k1': 0.123})
    '1=b, k1=0.123'
    """
    return sep.join('%s=%s' % (k, v) for k, v in d.items())


#class SeparatedClassifier


class ModelFactory:
    def __init__(self, model_class, *args, **kwd_args):
        self.model_class = model_class
        self.args = args
        self.kwd_args = kwd_args

    def __str__(self):
        pos_args = ', '.join(str(a) for a in self.args) if self.args else ''
        kwd_args = dict_tostring(self.kwd_args)
        delim = ', ' if pos_args and kwd_args else ''
        return self.model_class.__name__ + '(' + pos_args + delim + kwd_args + ')'

    def create(self):
        return self.model_class(*self.args, **self.kwd_args)


class IndependentMultiDimensionalClassifier:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y, *args, **kwargs):
        p = []
        for c in COVERAGE:
            p.append(self.models[c].predict(X))




if __name__ == '__main__':
    import doctest
    doctest.testmod()