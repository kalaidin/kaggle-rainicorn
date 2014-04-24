#!/usr/bin/env python3
"""Here I try to find out how sampling is performed

Metrics that show that sampling method is good:
* 25-50-90 percentiles -- must equal to the one from test distribution
* LastQuoted score -- must equal to benchmark from leaderboard (0.53793)
* additional metrics (for exploratory purposes): LastQuoted for each coverage

result:
/usr/bin/python3m /home/marat/kaggle-rainicorn/sampling_mist.py
** data has been read **

halting probability = 0.3
LastQuoted: 0.54932016617
A: 0.880
B: 0.889
C: 0.875
D: 0.906
E: 0.891
F: 0.876
G: 0.804

halting probability = 0.334693877551
LastQuoted: 0.537939778783
A: 0.875
B: 0.885
C: 0.868
D: 0.901
E: 0.887
F: 0.871
G: 0.800

halting probability = 0.4
LastQuoted: 0.516869568803
A: 0.867
B: 0.878
C: 0.860
D: 0.894
E: 0.880
F: 0.863
G: 0.791

"""

import doctest
from itertools import chain
from matplotlib.pyplot import plot, show, figure, legend, hist, title

import numpy as np
import pandas as pd
import sys
from commons import *


def calculate_clipped_length(length, prob_of_stay):
    """
    >>> calculate_clipped_length(10, 1)
    10
    >>> calculate_clipped_length(10, 0)
    2
    """
    if length <= 2:
        return length
    for new_len in range(2, length):
        if np.random.uniform() > prob_of_stay:
            return new_len
    return length


def sample_halt_with_probability(df, halt_prob):
    """Takes two transaction points then halts with a given probability
    Return (new_length:int, last_quoted:[int], actual_cov:[int]), weight for each customer
    weight for each customer sums up to 1.0"""
    for customer_id, group in df.groupby('customer_ID'):
        n = group.shape[0]
        for new_len, w in enumerate_samples(n - 1, 1 - halt_prob):
            last_quoted_coverage = group.iloc[new_len - 1][COVERAGE].values
            actual_coverage = group.iloc[-1][COVERAGE].values
            yield new_len, last_quoted_coverage, actual_coverage, w


doctest.testmod()

train = read_data('train.csv')
train_uniq_customers_count = train['customer_ID'].unique().shape[0]
test = read_data('test.csv')
print('** data has been read **')

for prob_halt in [1/3]:  # np.linspace(0.3, 0.4, num=2):
    print()
    print('halting probability =', prob_halt)

    results = [(nl * w, np.all(lq == ac) * w, (lq == ac) * w) for nl, lq, ac, w in sample_halt_with_probability(train, prob_halt)]

    train_length, lq, lq_by_coverage = list(zip(*results))
    train_length = np.array(train_length)
    test_length = test.groupby('customer_ID').count().values[:, 0]

    #hist(train_length, bins=10, label='train', normed=True)
    #hist(test_length, bins=10, alpha=0.5, label='test', normed=True)
    #legend()
    #title('Customer Transactions Count Distribution')
    #show()

    print('LastQuoted:', sum(lq) / train_uniq_customers_count)
    by_coverage = np.vstack(lq_by_coverage).sum(axis=0) / train_uniq_customers_count
    for i, c in enumerate(COVERAGE):
        print('%s: %.3f' % (c, by_coverage[i]))
