#!/usr/bin/env python3
import pickle
import datetime
import numpy as np
from commons import *


def weighted_score(actual, predicted, weight):
    assert isinstance(actual, np.ndarray)
    assert isinstance(predicted, np.ndarray)
    assert isinstance(weight, np.ndarray)

    corrects = 0
    total = 0
    for a, p, w in zip(actual, predicted, weight):
        if tuple(a) == tuple(p):
            corrects += w
        total += w
    return corrects / total


def eval_per_customer_model(model_class, *params, **kwd_params):
    corrects = 0.0
    total = 0.0
    for cv in range(CV_GROUPS_COUNT):
        with timer('cv group %d' % cv):
            with open(j(DATA_DIR, 'cv%02d_per-customer-train.pickle' % cv), 'rb') as f:
                dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

            model = model_class(*params, feature_names=dv.get_feature_names(), **kwd_params)
            model.fit(train_x, train_y, train_weights)

            with open(j(DATA_DIR, 'cv%02d_per-customer-test.pickle' % cv), 'rb') as f:
                _, test_customers, test_y, test_x, test_weights = pickle.load(f)

            test_p = model.predict(test_x)
            #w = np.ones(test_y.shape[0])
            corrects += (equal_rows(test_y, test_p) * test_weights).sum()
            total += test_weights.sum()

            train_p = model.predict(train_x)
            print('train cv group score:', weighted_score(train_y, train_p, train_weights))
            print('test cv group score:', weighted_score(test_y, test_p, test_weights))
    score = corrects / total
    print('Score for %s is %f' % (str(model), score))
    return score


def make_per_customer_submission(model_class, *params, **kwd_params):
    with open(j(DATA_DIR, 'per-customer-train.pickle'), 'rb') as f:
        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)

    model = model_class(*params, feature_names=dv.get_feature_names(), **kwd_params)
    model.fit(train_x, train_y, train_weights)

    with open(j(DATA_DIR, 'per-customer-test.pickle'), 'rb') as f:
        _, test_customers, test_y, test_x, test_weights = pickle.load(f)

    with open(j('submissions', '%s.csv' % datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")), 'w') as f:
        f.write('customer_ID,plan\n')
        for c, ps in zip(test_customers, model.predict(test_x)):
            f.write('%s,%s\n' % (c, ''.join(str(pp) for pp in ps)))


