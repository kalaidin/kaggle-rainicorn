#!/usr/bin/env python3
import numpy as np
from functools import lru_cache
from itertools import chain
from sklearn.feature_extraction import DictVectorizer
import pickle

from commons import *

FAKE_TARGET = [-1] * len(COVERAGE)
TAKE_PROB = 0.4

transformers = {}


def feature_each_coverage(func):
    """Creates a transformer for each coverage. Function must contain X symbol
    that will be replaced by coverage"""
    global transformers
    basic_name = func.__name__
    for c in COVERAGE:
        def f(data, coverage=c):
            return func(data, coverage=coverage)
        transformers[basic_name.replace('X', c)] = f
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
    global transformers
    transformers[func.__name__] = func
    return not_implemented_raiser(message='Use instead "%s" function' % func.__name__)


def _get_column_last_value(data, column):
    return data.tail(1)[column].values[0]


@feature_each_coverage
def last_X(data, coverage=None):
    return [categorical_pair('last_' + coverage, _get_column_last_value(data, coverage))]


#@feature
#@categorical
#def location(data):
#    return _get_column_last_value(data, 'location')


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


def make_features(data_grouped_by_customer, features_list=None):
    if features_list is None:
        features_list = sorted(transformers)
    print('features list is', features_list)
    trs = [f for fn, f in transformers.items() if fn in features_list]

    for x in data_grouped_by_customer:
        if len(x) == 2:
            customer, g = x
            w = 1
            target = FAKE_TARGET
        else:
            customer, g, w, target = x
        row = dict(chain(*(f(g) for f in trs)))
        yield customer, target, row, w


@lru_cache(maxsize=None)
def weight(current, total):
    """
    >>> abs(weight(0, 3) + weight(1, 3) + weight(2, 3) - 1) < 1e-10
    True
    """
    s = sum(TAKE_PROB**i for i in range(total))
    return TAKE_PROB**current / s


def slice_and_group(data):
    for customer, g in data.groupby('customer_ID'):
        target = g.iloc[-1][COVERAGE].values  # last row is target
        n = len(g)  # use all but last row
        for i in range(1, n):
            data_subset = g.iloc[0:i]
            w = weight(i - 1, n)
            yield customer, data_subset, w, target


def save(file_name, dict_vectorizer, customers, y, x, weights, savemat=True):
    with open(os.path.join(DATA_DIR, file_name + '.pickle'), 'wb') as f:
        pickle.dump((dict_vectorizer, customers, np.array(y), x, np.array(weights)), f)
    if savemat:
        header = tuple(COVERAGE) + ('weight',) + tuple(dict_vectorizer.get_feature_names())
        with open(os.path.join(DATA_DIR, file_name + '.csv'), 'w') as f:
            f.write(','.join(header))
            for c, w, i in zip(y, weights, range(x.shape[0])):
                row = tuple(c) + (w,) + tuple(x[i, :].todense())
                f.write(','.join(str(r) for r in row) + '\n')


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    dv = DictVectorizer()
    train_data = read_data('train.csv')
    train_customers, train_y, train_x, train_weights = zip(*make_features(slice_and_group(train_data)))
    train_x = dv.fit_transform(train_x)
    save('per-user-train', dv, train_customers, train_y, train_x, train_weights)

    test_data = read_data('test.csv')
    test_customers, test_y, test_x, test_weights = zip(*make_features(test_data.groupby('customer_ID')))
    test_x = dv.transform(test_x)
    save('per-user-test', dv, test_customers, test_y, test_x, test_weights)

    for cv_i, (train_raw, test_raw) in enumerate(cv(train_data)):
        dv = DictVectorizer()
        train_customers, train_y, train_x, train_weights = zip(*make_features(slice_and_group(train_raw)))
        train_x = dv.fit_transform(train_x)
        save('cv%02d_per-user-train' % cv_i, dv, train_customers, train_y, train_x, train_weights)

        test_customers, test_y, test_x, test_weights = zip(*make_features(slice_and_group(test_raw)))
        test_x = dv.transform(test_x)
        save('cv%02d_per-user-test' % cv_i, dv, test_customers, test_y, test_x, test_weights)


