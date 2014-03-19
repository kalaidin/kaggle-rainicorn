#!/usr/bin/env python3
import os
from random import random
import pandas as pd
from sklearn.cross_validation import KFold

CUSTOMER_ID = 'customer_ID'

DATA_DIR = 'data'
CV_DIR = 'cv'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
DELIM = ','

TERMINATION_PROB = 0.4

COVERAGE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


def read_data(file='train'):
    """file argument is from {'train', 'test'}"""
    file_path = TRAIN_FILE if file == 'train' else TEST_FILE
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
    for train_i, test_i in KFold(len(customers), shuffle=True):
        train_customers = {customers[i] for i in train_i}
        test_customers = {customers[i] for i in test_i}
        train = data[data[CUSTOMER_ID].isin(train_customers)]
        test = convert_to_test(data[data[CUSTOMER_ID].isin(test_customers)])
        yield train, test


def not_implemented_raiser(message=''):
    def f():
        raise NameError(message=message)
    return f


