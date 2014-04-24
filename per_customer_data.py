#!/usr/bin/env python3
"""This module splits data into cv-groups, makes each customer a row in a dataset. Features are created
by functions under @Feature decorator.

Consider features immutable. If you want to change something in a feature generating function try
adding new feature"""


import numpy as np
from itertools import chain
from sklearn.feature_extraction import DictVectorizer
import pickle
import sys

from commons import *

FAKE_TARGET = [-1] * len(COVERAGE)

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


def continuous_pair(feature_name, continuous_value):
    return feature_name, continuous_value


def normalize(mean=0, std_dev=1):
    def dec(func):
        def wrapped(*args, **kwargs):
            return (func(*args, **kwargs) - mean) / std_dev
        return wrapped
    return dec


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


@feature_each_coverage
def cont_last_X(data, coverage=None):
    return [continuous_pair('cont_last_' + coverage, _get_column_last_value(data, coverage))]


@feature_each_coverage
def viewed_X(data, coverage=None):
    return [categorical_pair('viewed_' + coverage, data[coverage].max())]


# this feature is no use, so it is excluded from all datasets
@feature
@categorical
def location(data):
    return _get_column_last_value(data, 'location')


@feature
@categorical
def location(data):
    return _get_column_last_value(data, 'state')


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
@continuous
@normalize(2, 0.72)
def n_car_age(data):
    return np.log(_get_column_last_value(data, 'car_age') + 1)


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
def cont_C_previous(data):
    return _get_column_last_value(data, 'C_previous')


@feature
@continuous
def duration_previous(data):
    return _get_column_last_value(data, 'duration_previous')


@feature
@continuous
def duration_previous(data):
    v = _get_column_last_value(data, 'duration_previous')
    return v if not np.isnan(v) else 0


@feature
@continuous
def cost(data):
    return _get_column_last_value(data, 'cost')


@feature
@categorical
def day_first_visit(data):
    return data.iloc[0]['day']


@feature
@categorical
def day_last_visit(data):
    return data.iloc[-1]['day']


def make_features(data_grouped_by_customer, features_list=None):
    if features_list is None:
        features_list = sorted(transformers)
    #print('features list is', features_list)
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


def slice_and_group(data):
    for customer, g in data.groupby('customer_ID'):
        target = g.iloc[-1][COVERAGE].values  # last row is target
        n = len(g)
        for new_len, w in enumerate_samples(n - 1, 1 - TERMINATION_PROB):
            data_subset = g.iloc[0:new_len]
            yield customer, data_subset, w, target


def save(file_name, dataset_name, dict_vectorizer, customers, y, x, weights, savemat=True):
    with open(j(DATA_DIR, dataset_name, file_name + '.pickle'), 'wb') as f:
        pickle.dump((dict_vectorizer, customers, np.array(y, dtype='int'), x, np.array(weights)), f)

    if savemat:
        header = tuple(COVERAGE) + ('weight',) + tuple(dict_vectorizer.get_feature_names())
        with open(j(DATA_DIR, dataset_name, file_name + '.csv'), 'w') as f:
            f.write(','.join(header) + '\n')
            for c, w, i in zip(y, weights, range(x.shape[0])):
                row = tuple(c) + (w,) + tuple(x[i, :].toarray()[0])
                f.write(','.join(str(r) for r in row) + '\n')


def create_dataset(dataset_name, features):
    print('Creating "%s" dataset with the following features: %s.' % (dataset_name, ', '.join(features)),
          file=sys.stderr)

    dv = DictVectorizer()
    train_data = read_data('train.csv')
    train_customers, train_y, train_x, train_weights = zip(*make_features(slice_and_group(train_data), features))
    train_x = dv.fit_transform(train_x)
    os.mkdir(j(DATA_DIR, dataset_name))
    save('per-customer-train', dataset_name, dv, train_customers, train_y, train_x, train_weights)

    test_data = read_data('test.csv')
    test_customers, test_y, test_x, test_weights = zip(*make_features(test_data.groupby('customer_ID'), features))
    test_x = dv.transform(test_x)
    save('per-customer-test', dataset_name, dv, test_customers, test_y, test_x, test_weights)

    for cv_i, (train_raw, test_raw) in enumerate(cv(train_data, CV_GROUPS_COUNT)):
        dv = DictVectorizer()
        train_customers, train_y, train_x, train_weights = zip(*make_features(slice_and_group(train_raw), features))
        train_x = dv.fit_transform(train_x)
        save('cv%02d_per-customer-train' % cv_i, dataset_name, dv, train_customers, train_y, train_x, train_weights)

        test_customers, test_y, test_x, test_weights = zip(*make_features(slice_and_group(test_raw), features))
        test_x = dv.transform(test_x)
        save('cv%02d_per-customer-test' % cv_i, dataset_name, dv, test_customers, test_y, test_x, test_weights)


def main():
    per_customer_cat2_features = ['age_youngest',
                                  'duration_previous',
                                  'age_oldest',
                                  'car_value',
                                  'married_couple',
                                  'group_size',
                                  'risk_factor',
                                  'homeowner',
                                  'cost',
                                  'C_previous',
                                  'last_A',
                                  'last_B',
                                  'last_C',
                                  'last_D',
                                  'last_E',
                                  'last_F',
                                  'last_G',
                                  'viewed_A',
                                  'viewed_B',
                                  'viewed_C',
                                  'viewed_D',
                                  'viewed_E',
                                  'viewed_F',
                                  'viewed_G',
                                  'order',
                                  'day_first_visit',
                                  'day_last_visit',
                                  'state',
                                  'n_car_age']
    create_dataset('per_customer_cat2_features', per_customer_cat2_features)

    per_customer_cat_features = ['age_youngest',
                                 'duration_previous',
                                 'age_oldest',
                                 'car_value',
                                 'married_couple',
                                 'car_age',
                                 'group_size',
                                 'risk_factor',
                                 'homeowner',
                                 'cost',
                                 'C_previous',
                                 'last_A',
                                 'last_B',
                                 'last_C',
                                 'last_D',
                                 'last_E',
                                 'last_F',
                                 'last_G',
                                 'order']
    create_dataset('per-customer-cat', per_customer_cat_features)

    per_customer_cont_features = ['age_youngest',
                                  'duration_previous',
                                  'age_oldest',
                                  'car_value',
                                  'married_couple',
                                  'car_age',
                                  'group_size',
                                  'risk_factor',
                                  'homeowner',
                                  'cost',
                                  'cont_C_previous',
                                  'cont_last_A',
                                  'cont_last_B',
                                  'cont_last_C',
                                  'cont_last_D',
                                  'cont_last_E',
                                  'cont_last_F',
                                  'cont_last_G',
                                  'order']
    create_dataset('per-customer-cont', per_customer_cont_features)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()