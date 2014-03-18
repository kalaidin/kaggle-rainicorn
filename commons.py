#!/usr/bin/env python3

import doctest
from itertools import chain
from random import random
from time import time
from matplotlib.pyplot import plot, show, figure, legend, hist, title

import numpy as np
import pandas as pd
import sys
from sklearn.cross_validation import KFold
from sklearn.feature_extraction import DictVectorizer


CUSTOMER_ID = 'customer_ID'

DATA_DIR = 'data'
CV_DIR = 'cv'
TRAIN_FILE = DATA_DIR + '/train.csv'
TEST_FILE = DATA_DIR + '/test.csv'
DELIM = ','

TERMINATION_PROB = 0.4

COVERAGE = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

_FEATURE_PREFIX = '_feature_'


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


def feature_each_coverage(func):
    basic_name = func.__name__
    for c in COVERAGE:
        def f(data, coverage=c):
            return func(data, coverage=coverage)
        globals()[_FEATURE_PREFIX + basic_name.replace('X', c)] = f
    return not_implemented_raiser()


def categorical_pair(feature_name, categorical_value):
    key = '%s:%s' % (feature_name, categorical_value)
    return key, 1.0


def categorical(func):
    def wrapped(*args):
        r = func(*args)
        var = func.__name__
        return [categorical_pair(var, r)]
    wrapped.__name__ = func.__name__
    return wrapped


def continuous(func):
    def wrapped(*args):
        r = func(*args)
        var = func.__name__
        return [(var, r)]
    wrapped.__name__ = func.__name__
    return wrapped


def feature(func):
    """Decorator. Decorated functions must return a list of pairs [(variable1,value1), ...] where values are real"""
    new_func_name = _FEATURE_PREFIX + func.__name__
    func.__name__ = new_func_name
    globals()[new_func_name] = func
    return not_implemented_raiser(message='Use instead "%s" function' % new_func_name)


def _get_column_last_value(data, column):
    return data.tail(1)[column].values[0]


@feature_each_coverage
def current_X(data, coverage=None):
    return [categorical_pair('current_' + coverage, _get_column_last_value(data, coverage))]


@feature
@categorical
def location(data):
    return _get_column_last_value(data, 'location')


@feature
@categorical
def group_size(data):
    return _get_column_last_value(data, 'group_size')


@feature
@categorical
def homeowner(data):
    return _get_column_last_value(data, 'homeowner')


@feature
@continuous
def order(data):
    return len(data)


@feature
@continuous
def car_age(data):
    return _get_column_last_value(data, 'car_age')


@feature
@categorical
def car_value(data):
    return _get_column_last_value(data, 'car_value')


@feature
@categorical
def risk_factor(data):
    return _get_column_last_value(data, 'risk_factor')


@feature
@continuous
def age_oldest(data):
    return _get_column_last_value(data, 'age_oldest')


@feature
@continuous
def age_youngest(data):
    return _get_column_last_value(data, 'age_youngest')


@feature
@continuous
def married_couple(data):
    return _get_column_last_value(data, 'married_couple')


@feature
@categorical
def C_previous(data):
    return _get_column_last_value(data, 'C_previous')


@feature
@continuous
def duration_previous(data):
    return _get_column_last_value(data, 'duration_previous')


@feature
@continuous
def cost(data):
    return _get_column_last_value(data, 'cost')


def feature_transformer(feature_name):
    return globals()[_FEATURE_PREFIX + feature_name]


def make_features(data, features_list=None):
    if features_list is None:
        features_list = [f[len(_FEATURE_PREFIX):] for f in sorted(globals()) if f.startswith(_FEATURE_PREFIX)]
    print('features list is', features_list)
    transformers = {f: feature_transformer(f) for f in features_list}

    for customer, g in data.groupby(CUSTOMER_ID):
        for i in range(1, len(g)):
            data_subset = g.iloc[0:i]
            row = dict(chain(*(transformers[f](data_subset) for f in features_list)))
            yield row


def make_targets(data, covarage):
    for customer, g in data.groupby(CUSTOMER_ID):
        for i in range(1, len(g)):
            data_subset = g.iloc[i + 1]
            yield data_subset[covarage].values[0]

all_data = read_data('train')[:100000]
t = time()
d = tuple(make_features(all_data, ['cost', 'homeowner', 'current_A']))
print('Ellapsed %.1f seconds' % (time() - t))
print(len(d))
#
#import cProfile
#cProfile.run("tuple(make_features(all_data, ['cost', 'homeowner']))")
#
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GephiOutput, GraphvizOutput
#
#with PyCallGraph(output=GephiOutput(output_file='filter_exclude')):
#    tuple(make_features(all_data, ['cost', 'homeowner']))