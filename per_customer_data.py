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

per_customer_1_features = ['age_oldest',
                           'age_youngest',
                           'age_difference',
                           'age_prop'
                           'age_youngest_ranged',
                           'age_oldest_ranged',
                           'duration_previous',
                           'duration_previous_safe',
                           'duration_previous_ranged',
                           'married_couple',
                           'group_size',
                           'homeowner',
                           'C_previous',
                           'last_A',
                           'last_B',
                           'last_C',
                           'last_D',
                           'last_E',
                           'last_F',
                           'last_G',
                           'cont_C_previous',
                           'cont_last_A',
                           'cont_last_B',
                           'cont_last_C',
                           'cont_last_D',
                           'cont_last_E',
                           'cont_last_F',
                           'cont_last_G',
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
                           'car_age',
                           'car_age_norm',
                           'car_value',
                           'risk_factor',
                           'cost',
                           'cost_norm']


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


def ranged(thresholds, labels=None):
    def dec(func):
        def wrapped(*args, **kwargs):
            v = func(*args, **kwargs)
            return ranged_value(v, thresholds, labels)
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
def car_age_norm(data):
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
def age_difference(data):
    return _get_column_last_value(data, 'age_oldest') - _get_column_last_value(data, 'age_youngest')


@feature
@continuous
def age_prop(data):
    return _get_column_last_value(data, 'age_youngest') / _get_column_last_value(data, 'age_oldest')


@feature
@categorical
@ranged([38, 73], ['young', 'middle', 'old'])
def age_oldest_cat(data):
    return _get_column_last_value(data, 'age_oldest')


@feature
@categorical
@ranged([38, 73], ['young', 'middle', 'old'])
def age_youngest_cat(data):
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
def duration_previous_safe(data):
    v = _get_column_last_value(data, 'duration_previous')
    return v if not np.isnan(v) else 0


@feature
@categorical
def duration_previous_ranged(data):
    v = _get_column_last_value(data, 'duration_previous')
    if np.isnan(v):
        return 'NaN'
    return ranged_value(v, [1, 5, 11, 15], labels=['0', '1-4', '5-10', '11-14', '15'])


@feature
@continuous
def cost(data):
    return _get_column_last_value(data, 'cost')


@feature
@continuous
@normalize(634, 43)
def cost_norm(data):
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
    create_dataset('per-customer-1', per_customer_1_features)
    create_dataset('per-customer-cat', per_customer_cat_features)
    create_dataset('per-customer-cont', per_customer_cont_features)


if __name__ == '__main__':
    import doctest
    if doctest.testmod().failed > 0:
        print('tests failed!', file=sys.stderr)

    main()