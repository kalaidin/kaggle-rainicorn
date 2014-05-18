#!/usr/bin/env python3
import pickle
import datetime
import numpy as np
from scipy.sparse import hstack, csc_matrix
from sklearn import clone
from commons import *


def weighted_score(actual, predicted, sample_weight=None):
    """
    >>> import numpy as np
    >>> a = np.array([[1, 1], [1, 2]])
    >>> p = np.array([[1, 0], [1, 2]])
    >>> w = np.array([0.7, 0.3])
    >>> '%0.1f' % weighted_score(a, p, w)
    '0.3'
    """
    assert isinstance(actual, np.ndarray)
    assert isinstance(predicted, np.ndarray)
    assert (sample_weight is None) or isinstance(sample_weight, np.ndarray)
    if sample_weight is None:
        sample_weight = np.ones(actual.shape)

    corrects = actual == predicted if len(actual.shape) == 1 else np.all(actual == predicted, axis=1)
    return np.sum(corrects * sample_weight) / sum(sample_weight)


def get_feature_indexi(all_features, include_list, strict_eq=False):
    """
    >>> get_feature_indexi(['a', 'b', 'c:1.0'], ['b', 'c'])
    (1, 2)
    >>> get_feature_indexi(['a:1.0', 'c:0.0', 'c:1.0'], ['a:1.0', 'c:1.0'], strict_eq=True)
    (0, 2)"""

    if strict_eq:
        included = filter(lambda x: x[1] in include_list, enumerate(all_features))
    else:
        included = filter(lambda x: x[1].split(':')[0] in include_list, enumerate(all_features))

    return_indexi = list(zip(*included))[0]
    return return_indexi


def multiply_row_wise(mat):
    return csc_matrix(np.expm1(mat.log1p().sum(axis=1)))


def creating_interacting_matrix(matrix, column_names, include_variables, interacting_variables):
    first_order_indexi = np.array(get_feature_indexi(column_names, include_variables), dtype='int')
    multiorder_indexi = [np.array(get_feature_indexi(column_names, interaction, strict_eq=True), dtype='int') for interaction in interacting_variables]
    csc = matrix.tocsc()
    return hstack([csc[:, first_order_indexi]] + [multiply_row_wise(csc[:, ii]) for ii in multiorder_indexi]).tocsr()


def eval_per_customer_model(dataset_name, estimator, x_transformation=identity, y_transformation=identity,
                            include_variables=None, interacting_variables=[]):
    """ Order of transformations on X:
    1. apply include_variables and exclude_variables and interacting_variables
    2. apply x_transformation
    """
    corrects = 0.0
    total = 0.0
    for cv in range(CV_GROUPS_COUNT):
        with timer('cv group %d' % cv):
            with open(j(DATA_DIR, dataset_name, 'cv%02d_per-customer-train.pickle' % cv), 'rb') as f:
                dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

            choose_columnns = lambda x: creating_interacting_matrix(x, dv.get_feature_names(),
                                                                    include_variables, interacting_variables)
            x_all_transforms = lambda x: x_transformation(choose_columnns(x))

            model = clone(estimator)
            try:
                model.fit(x_all_transforms(train_x), y_transformation(train_y), sample_weight=train_weights)
            except TypeError:
                print("%s doesn't support `sample_weight`. Ignoring it." % str(model))
                model.fit(x_all_transforms(train_x), y_transformation(train_y))

            with open(j(DATA_DIR, dataset_name, 'cv%02d_per-customer-test.pickle' % cv), 'rb') as f:
                _, test_customers, test_y, test_x, test_weights = pickle.load(f)

            test_p = model.predict(x_all_transforms(test_x))
            corrects += (equal_rows(y_transformation(test_y), test_p) * test_weights).sum()
            total += test_weights.sum()

            train_p = model.predict(x_all_transforms(train_x))
            print('train cv group score:', weighted_score(y_transformation(train_y), train_p, train_weights))
            print('test cv group score:', weighted_score(y_transformation(test_y), test_p, test_weights))
    score = corrects / total
    print('Score for %s is %f' % (str(model), score))
    return score


def get_feature_names(dataset_name):
    with open(j(DATA_DIR, dataset_name, 'per-customer-train.pickle'), 'rb') as f:
        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)
    return dv.get_feature_names()


def make_per_customer_submission(dataset_name, estimator, x_transformation=identity, y_transformation=identity,
                            include_variables=None):
    """BROKEN FOR NOW!"""
    with open(j(DATA_DIR, dataset_name, 'per-customer-train.pickle'), 'rb') as f:
        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

    choose_columnns = lambda x: x[:, np.array(get_feature_indexi(dv.get_feature_names(), include_variables), dtype='int')]
    x_all_transforms = lambda x: x_transformation(choose_columnns(x))

    model = clone(estimator)
    try:
        model.fit(x_all_transforms(train_x), y_transformation(train_y), sample_weight=train_weights)
    except TypeError:
        print("%s doesn't support `sample_weight`. Ignoring it." % str(model))
        model.fit(x_all_transforms(train_x), y_transformation(train_y))

    with open(j(DATA_DIR, dataset_name, 'per-customer-test.pickle'), 'rb') as f:
        _, test_customers, test_y, test_x, test_weights = pickle.load(f)

    with open(j('submissions', '%s.csv' % datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")), 'w') as f:
        f.write('customer_ID,plan\n')
        for c, ps in zip(test_customers, model.predict(x_all_transforms(test_x))):
            f.write('%s,%s\n' % (c, ''.join(str(pp) for pp in ps)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

