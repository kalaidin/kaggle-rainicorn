#!/usr/bin/env python3
"""Here I try to find out how sampling is performed

Metrics that show that sampling method is good:
* 25-50-90 percentiles -- must equal to the one from test distribution
* LastQuoted score -- must equal to benchmark from leaderboard (0.53793)
* additional metrics (for exploratory purposes): LastQuoted for each coverage
"""
import doctest
from itertools import chain
from matplotlib.pyplot import plot, show, figure, legend, hist, title

import numpy as np
import pandas as pd
import sys
from commons import *


def sample_halt_with_probability(halt_prob, lengths):
    """Takes two transaction points then halts with a given probability
    Return new lengths"""
    m = 1.0
    p_stay = 1 - halt_prob
    max_count = np.max(lengths) - 1
    probs = [p_stay * m ** i for i in range(max_count)]
    cutted_lengths = []
    for full_len in lengths:
        a = [np.random.uniform() < p for kk, p in zip(range(2, full_len - 1), probs)]
        new_len = full_len - 1 if False not in a else 2 + a.index(False)
        cutted_lengths.append(new_len)
    return cutted_lengths




train = read_data('train.csv')
test = read_data('test.csv')

train_length = train.groupby('customer_ID').count().values[:, 0]
test_length = test.groupby('customer_ID').count().values[:, 0]

hist(train_length - 1, bins=10, label='train')
hist(test_length, bins=10, alpha=0.5, label='test')
legend()
title('Customer Transactions Count Distribution')
show()


new_lengths = sample_halt_with_probability(0.5, train_length)
hist(new_lengths, label='simulated lambda=' + str(p_stay), alpha=0.5)
hist(np.hstack((test_length,) * 2), alpha=0.5, label='test distribution')
legend()
title('Customer Transactions Count Distribution clipped by increasing random')
show()



#d = train[train.shopping_pt != 1].groupby('car_value').aggregate({c: np.mean for c in COVERAGE})
#plot(d)
#show()


train['hour'] = train['time'].map(lambda x: x.split(':')[0])
hourly = train[train.shopping_pt == 1].groupby('hour')[COVERAGE].agg(np.mean)
for c in hourly.columns:
    plot(hourly[c], label=c)
legend(loc='auto')
show()


sys.exit(0)






def changed(data_points):
    a = np.array(data_points)
    return int(a[0] != a[-1])


def comp(data_points):
    """Return 1 if coverage is increased, -1 if decreased, 0 if stayed the same"""
    a = np.array(data_points)
    old = a[0]
    new = a[-1]
    if old == new:
        return 0
    elif new < old:
        return -1
    else:
        return 1


def incr(data_points):
    """Return delta"""
    a = np.array(data_points)
    old = a[0]
    new = a[-1]
    return new - old


def for_all_coverages(func):
    return {c: func for c in COVERAGE}

changes_by_customer = train.groupby('customer_ID').aggregate(for_all_coverages(incr)).groupby(COVERAGE).size().reset_index()
changes_by_customer.rename(columns={changes_by_customer.columns[-1]: 'size'}, inplace=True)
print(changes_by_customer.sort('size', ascending=0).to_string(index_names=False))

changes_by_session = train.groupby('customer_ID').aggregate(for_all_coverages(comp)).groupby(COVERAGE).size().reset_index()
changes_by_session.rename(columns={changes_by_customer.columns[-1]: 'size'}, inplace=True)

train.loc[train['record_type']==1 ].count()


def iter_pairs(rows):
    """
    >>> tuple(iter_pairs(range(3)))
    ((0, 1), (1, 2))
    """
    if not rows:
        raise GeneratorExit()
    i = iter(rows)
    prev = next(i)
    for line in i:
        yield (prev, line)
        prev = line

doctest.testmod()

#delta = []
#for customer_ID, g in train.groupby('customer_ID'):
#    covs = g[COVERAGE].values
#    for i in range(1, covs.shape[0]):
#        delta.append(tuple(covs[i-1] - covs[i]))
#
#delta = np.array(delta)
#all_changes = pd.DataFrame(delta, columns=COVERAGE)
#ppp = all_changes.groupby(COVERAGE).size().reset_index().sort(0, ascending=0).to_string()
#
#with open('all_changes.txt', 'wb') as c:
#    c.write(bytes(ppp, encoding='utf-8'))