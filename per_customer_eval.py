#!/usr/bin/env python3
import pickle
from commons import *
from per_customer_models import LastQuoted


def weighted_score(actual, predicted, weight):
    corrects = 0
    total = 0
    for a, p, w in zip(actual, predicted, weight):
        if tuple(a) == tuple(p):
            corrects += w
        total += w
    return corrects / total

#corrects = 0.0
#total = 0.0
#for cv in range(CV_GROUPS_COUNT):
#    with open(j(DATA_DIR, 'cv%02d_train.pickle' % cv), 'rb') as f:
#        dv, train_customers, train_y, train_x, train_weights = pickle.load(f)
#    lq = LastQuoted(dv.get_feature_names())
#
#    with open(j(DATA_DIR, 'cv%02d_test.pickle' % cv), 'rb') as f:
#        _, test_customers, test_y, test_x, test_weights = pickle.load(f)
#
#    pred = lq.predict(test_x)
#    for actual, predicted, w in zip(test_y, pred, test_weights):
#        if tuple(actual) == tuple(predicted):
#            corrects += 1
#        total += 1
#
#    print(weighted_score(train_y, lq.predict(train_x), train_weights))
#
#print('Score', corrects / total)

with open(j(DATA_DIR, 'test.pickle'), 'rb') as f:
    dv, test_customers, test_y, test_x, test_weights = pickle.load(f)

lq = LastQuoted(dv.get_feature_names())
with open(j('submissions', '1.csv'), 'w') as f:
    f.write('customer_ID,plan\n')
    for c, ps in zip(test_customers, lq.predict(test_x)):
        f.write('%s,%s\n' % (c, ''.join(str(pp) for pp in ps)))

