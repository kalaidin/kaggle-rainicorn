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
DELIM = ','

TERMINATION_PROB = 1 / 3

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


def read_data(file_name):
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


def cv(data, cv_groups_count=3):
    """Converts pandas DataFrame into KFold-cv groups, where groups
    created wrt customer_ID"""
    assert isinstance(data, pd.core.frame.DataFrame)
    customers = tuple(set(data[CUSTOMER_ID].values))
    for train_i, test_i in KFold(len(customers), shuffle=True, n_folds=cv_groups_count):
        train_customers = {customers[i] for i in train_i}
        test_customers = {customers[i] for i in test_i}
        train = data[data[CUSTOMER_ID].isin(train_customers)]
        #test = convert_to_test(data[data[CUSTOMER_ID].isin(test_customers)])
        test = data[data[CUSTOMER_ID].isin(test_customers)]
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
    if len(mat1.shape) == 1:
        return mat1 == mat2
    else:
        return np.all(mat1 == mat2, axis=1)


def dict_tostring(d, sep=', '):
    """
    >>> dict_tostring({1: 'b', 'k1': 0.123})
    '1=b, k1=0.123'
    """
    return sep.join('%s=%s' % (k, v) for k, v in d.items())


def enumerate_samples(length_without_last_row, prob_of_stay):
    """yields new_length and its probability
    >>> a = list(enumerate_samples(4, 0.66))
    >>> a
    [(2, 0.33999999999999997), (3, 0.2244), (4, 0.43560000000000004)]
    >>> sum(list(zip(*a))[1])
    1.0"""

    assert length_without_last_row >= 2
    for new_len in range(2, length_without_last_row):
        yield new_len, (1 - prob_of_stay) * prob_of_stay ** (new_len - 2)
    yield length_without_last_row, prob_of_stay ** (length_without_last_row - 2)


def identity(value):
    return value


if __name__ == '__main__':
    import doctest
    doctest.testmod()