#!/usr/bin/env python3
import pickle
import datetime
import numpy as np
from sklearn import clone
from commons import *


def weighted_score(actual, predicted, weight):
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
    assert isinstance(weight, np.ndarray)

    corrects = actual == predicted if len(actual.shape) == 1 else np.all(actual == predicted, axis=1)
    return np.sum(corrects * weight) / sum(weight)


def eval_per_customer_model(dataset_name, estimator, x_transformation=identity, y_transformation=identity):
    corrects = 0.0
    total = 0.0
    for cv in range(CV_GROUPS_COUNT):
        with timer('cv group %d' % cv):
            with open(j(DATA_DIR, dataset_name, 'cv%02d_per-customer-train.pickle' % cv), 'rb') as f:
                dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

            model = clone(estimator)
            try:
                model.fit(x_transformation(train_x), y_transformation(train_y), sample_weight=train_weights)
            except TypeError:
                print("%s doesn't support `sample_weight`. Ignoring it." % str(model))
                model.fit(x_transformation(train_x), y_transformation(train_y))

            with open(j(DATA_DIR, dataset_name, 'cv%02d_per-customer-test.pickle' % cv), 'rb') as f:
                _, test_customers, test_y, test_x, test_weights = pickle.load(f)

            test_p = model.predict(x_transformation(test_x))
            corrects += (equal_rows(y_transformation(test_y), test_p) * test_weights).sum()
            total += test_weights.sum()

            train_p = model.predict(x_transformation(train_x))
            print('train cv group score:', weighted_score(y_transformation(train_y), train_p, train_weights))
            print('test cv group score:', weighted_score(y_transformation(test_y), test_p, test_weights))
    score = corrects / total
    print('Score for %s is %f' % (str(model), score))
    return score


def get_feature_names(dataset_name):
    with open(j(DATA_DIR, dataset_name, 'per-customer-train.pickle'), 'rb') as f:
        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)
    return dv.get_feature_names()


def make_per_customer_submission(dataset_name, model_class, *params, **kwd_params):
    with open(j(DATA_DIR, dataset_name, 'per-customer-train.pickle'), 'rb') as f:
        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

    model = model_class(*params, feature_names=dv.get_feature_names(), **kwd_params)
    model.fit(train_x, train_y, train_weights)

    with open(j(DATA_DIR, dataset_name, 'per-customer-test.pickle'), 'rb') as f:
        _, test_customers, test_y, test_x, test_weights = pickle.load(f)

    with open(j('submissions', '%s.csv' % datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), 'w') as f:
        f.write('customer_ID,plan\n')
        for c, ps in zip(test_customers, model.predict(test_x)):
            f.write('%s,%s\n' % (c, ''.join(str(pp) for pp in ps)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

