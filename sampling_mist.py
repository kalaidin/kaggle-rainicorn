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

halting probability = 0.302040816327
LastQuoted: 0.548155325795
A: 0.880
B: 0.889
C: 0.874
D: 0.906
E: 0.890
F: 0.877
G: 0.804

halting probability = 0.304081632653
LastQuoted: 0.546629694152
A: 0.879
B: 0.888
C: 0.873
D: 0.905
E: 0.889
F: 0.876
G: 0.804

halting probability = 0.30612244898
LastQuoted: 0.546918327166
A: 0.880
B: 0.888
C: 0.873
D: 0.905
E: 0.890
F: 0.875
G: 0.804

halting probability = 0.308163265306
LastQuoted: 0.547021410385
A: 0.878
B: 0.888
C: 0.873
D: 0.905
E: 0.890
F: 0.874
G: 0.804

halting probability = 0.310204081633
LastQuoted: 0.545990578194
A: 0.878
B: 0.888
C: 0.872
D: 0.905
E: 0.890
F: 0.874
G: 0.804

halting probability = 0.312244897959
LastQuoted: 0.544341246688
A: 0.878
B: 0.888
C: 0.873
D: 0.905
E: 0.890
F: 0.875
G: 0.803

halting probability = 0.314285714286
LastQuoted: 0.544289705079
A: 0.878
B: 0.888
C: 0.872
D: 0.904
E: 0.890
F: 0.875
G: 0.802

halting probability = 0.316326530612
LastQuoted: 0.543227947922
A: 0.878
B: 0.887
C: 0.872
D: 0.903
E: 0.889
F: 0.875
G: 0.802

halting probability = 0.318367346939
LastQuoted: 0.541021967034
A: 0.878
B: 0.887
C: 0.871
D: 0.903
E: 0.889
F: 0.873
G: 0.801

halting probability = 0.320408163265
LastQuoted: 0.542258965663
A: 0.877
B: 0.886
C: 0.871
D: 0.904
E: 0.888
F: 0.873
G: 0.802

halting probability = 0.322448979592
LastQuoted: 0.541372449979
A: 0.877
B: 0.887
C: 0.870
D: 0.904
E: 0.888
F: 0.873
G: 0.802

halting probability = 0.324489795918
LastQuoted: 0.54035192611
A: 0.877
B: 0.886
C: 0.870
D: 0.903
E: 0.888
F: 0.873
G: 0.802

halting probability = 0.326530612245
LastQuoted: 0.538259336763
A: 0.876
B: 0.885
C: 0.870
D: 0.902
E: 0.887
F: 0.872
G: 0.801

halting probability = 0.328571428571
LastQuoted: 0.538001628715
A: 0.875
B: 0.886
C: 0.869
D: 0.902
E: 0.888
F: 0.872
G: 0.800

halting probability = 0.330612244898
LastQuoted: 0.538372728304
A: 0.875
B: 0.885
C: 0.870
D: 0.902
E: 0.886
F: 0.872
G: 0.800

halting probability = 0.332653061224
LastQuoted: 0.538754136214
A: 0.876
B: 0.885
C: 0.869
D: 0.903
E: 0.887
F: 0.872
G: 0.800

halting probability = 0.334693877551
LastQuoted: 0.537939778783
A: 0.875
B: 0.885
C: 0.868
D: 0.901
E: 0.887
F: 0.871
G: 0.800

halting probability = 0.336734693878
LastQuoted: 0.536754321764
A: 0.874
B: 0.886
C: 0.870
D: 0.902
E: 0.887
F: 0.871
G: 0.799

halting probability = 0.338775510204
LastQuoted: 0.535527631457
A: 0.874
B: 0.884
C: 0.868
D: 0.901
E: 0.885
F: 0.871
G: 0.798

halting probability = 0.340816326531
LastQuoted: 0.536537847004
A: 0.875
B: 0.885
C: 0.869
D: 0.901
E: 0.887
F: 0.870
G: 0.798

halting probability = 0.342857142857
LastQuoted: 0.5344658743
A: 0.874
B: 0.884
C: 0.866
D: 0.900
E: 0.885
F: 0.870
G: 0.799

halting probability = 0.344897959184
LastQuoted: 0.531703244029
A: 0.874
B: 0.884
C: 0.867
D: 0.900
E: 0.885
F: 0.869
G: 0.798

halting probability = 0.34693877551
LastQuoted: 0.53408446639
A: 0.874
B: 0.884
C: 0.868
D: 0.901
E: 0.886
F: 0.870
G: 0.799

halting probability = 0.348979591837
LastQuoted: 0.533507200363
A: 0.874
B: 0.884
C: 0.867
D: 0.901
E: 0.885
F: 0.870
G: 0.798

halting probability = 0.351020408163
LastQuoted: 0.531393994372
A: 0.873
B: 0.883
C: 0.867
D: 0.900
E: 0.885
F: 0.869
G: 0.798

halting probability = 0.35306122449
LastQuoted: 0.53089919492
A: 0.873
B: 0.883
C: 0.867
D: 0.899
E: 0.885
F: 0.869
G: 0.797

halting probability = 0.355102040816
LastQuoted: 0.532600068035
A: 0.872
B: 0.883
C: 0.867
D: 0.899
E: 0.886
F: 0.868
G: 0.798

halting probability = 0.357142857143
LastQuoted: 0.531744477317
A: 0.872
B: 0.883
C: 0.866
D: 0.899
E: 0.885
F: 0.869
G: 0.799

halting probability = 0.359183673469
LastQuoted: 0.529136471874
A: 0.871
B: 0.882
C: 0.865
D: 0.899
E: 0.883
F: 0.866
G: 0.797

halting probability = 0.361224489796
LastQuoted: 0.530126070777
A: 0.872
B: 0.883
C: 0.865
D: 0.899
E: 0.884
F: 0.868
G: 0.797

halting probability = 0.363265306122
LastQuoted: 0.529940520983
A: 0.872
B: 0.882
C: 0.865
D: 0.898
E: 0.884
F: 0.868
G: 0.798

halting probability = 0.365306122449
LastQuoted: 0.52884783886
A: 0.871
B: 0.882
C: 0.866
D: 0.899
E: 0.883
F: 0.868
G: 0.796

halting probability = 0.367346938776
LastQuoted: 0.526476924821
A: 0.872
B: 0.881
C: 0.864
D: 0.898
E: 0.883
F: 0.867
G: 0.795

halting probability = 0.369387755102
LastQuoted: 0.525518250884
A: 0.871
B: 0.880
C: 0.863
D: 0.897
E: 0.881
F: 0.866
G: 0.796

halting probability = 0.371428571429
LastQuoted: 0.526425383212
A: 0.872
B: 0.881
C: 0.864
D: 0.898
E: 0.884
F: 0.868
G: 0.795

halting probability = 0.373469387755
LastQuoted: 0.52501314311
A: 0.871
B: 0.881
C: 0.864
D: 0.898
E: 0.882
F: 0.867
G: 0.794

halting probability = 0.375510204082
LastQuoted: 0.524487418693
A: 0.870
B: 0.881
C: 0.863
D: 0.897
E: 0.882
F: 0.866
G: 0.794

halting probability = 0.377551020408
LastQuoted: 0.525806883897
A: 0.870
B: 0.881
C: 0.863
D: 0.898
E: 0.882
F: 0.867
G: 0.795

halting probability = 0.379591836735
LastQuoted: 0.523569978043
A: 0.869
B: 0.880
C: 0.862
D: 0.897
E: 0.882
F: 0.864
G: 0.793

halting probability = 0.381632653061
LastQuoted: 0.524260635611
A: 0.871
B: 0.880
C: 0.863
D: 0.897
E: 0.881
F: 0.866
G: 0.794

halting probability = 0.383673469388
LastQuoted: 0.522497912565
A: 0.869
B: 0.880
C: 0.862
D: 0.897
E: 0.882
F: 0.865
G: 0.794

halting probability = 0.385714285714
LastQuoted: 0.522869012154
A: 0.869
B: 0.879
C: 0.862
D: 0.896
E: 0.880
F: 0.865
G: 0.794

halting probability = 0.387755102041
LastQuoted: 0.521312455545
A: 0.868
B: 0.880
C: 0.862
D: 0.896
E: 0.881
F: 0.865
G: 0.794

halting probability = 0.389795918367
LastQuoted: 0.519642507396
A: 0.868
B: 0.879
C: 0.860
D: 0.895
E: 0.880
F: 0.864
G: 0.793

halting probability = 0.391836734694
LastQuoted: 0.519735282293
A: 0.869
B: 0.879
C: 0.861
D: 0.895
E: 0.880
F: 0.865
G: 0.791

halting probability = 0.39387755102
LastQuoted: 0.519745590615
A: 0.868
B: 0.879
C: 0.861
D: 0.895
E: 0.880
F: 0.864
G: 0.792

halting probability = 0.395918367347
LastQuoted: 0.519003391438
A: 0.869
B: 0.880
C: 0.861
D: 0.895
E: 0.880
F: 0.863
G: 0.792

halting probability = 0.397959183673
LastQuoted: 0.518920924863
A: 0.868
B: 0.879
C: 0.860
D: 0.895
E: 0.881
F: 0.863
G: 0.793

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