#!/usr/bin/env python3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
import sys
from functools import partial

from per_customer_eval import *
from per_customer_models import *

eval = partial(eval_per_customer_model, 'per-customer-cat', exclude_variables=[])
#features = partial(get_feature_names, 'per-customer-cat')

#LastQuoted
#eval(LastQuoted(feature_names=features()))
# Score for LastQuoted() is 0.537753
#A: 0.875
#B: 0.885
#C: 0.868
#D: 0.901
#E: 0.887
#F: 0.871
#G: 0.800

#eval_per_customer_model(LinearSVC(C=0.00001), y_transformation=lambda y: y[:, 0])
#A: 0.874905 (3 min)
#eval_per_customer_model(LinearSVC(C=0.0001), y_transformation=lambda y: y[:, 0])
#A: 0.874984 (20 min)
#eval_per_customer_model(SVC(kernel='linear', C=0.00001), y_transformation=lambda y: y[:, 0])

#eval_per_customer_model(SVC(kernel='rbf', C=0.00001), y_transformation=lambda y: y[:, 0])
#A: 0.72 (18 hours)

# same performance as LastQuoted
min_list = ['C_previous',
            'last_A',
            'last_B',
            'last_C',
            'last_D',
            'last_E',
            'last_F',
            'last_G']
eval(EachTaskIndependently(SGDClassifier(n_iter=5, shuffle=True)), include_variables=min_list)


#OneVsRestClassifier()
