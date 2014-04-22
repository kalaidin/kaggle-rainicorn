#!/usr/bin/env python3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
import sys
from functools import partial

from per_customer_eval import *
from per_customer_models import *

eval = partial(eval_per_customer_model, 'per-customer-cat')
features = partial(get_feature_names, 'per-customer-cat')

# LastQuoted
# eval(LastQuoted(feature_names=features()))
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


#eval_per_customer_model(LastQuoted)
#eval_per_customer_model(EachTaskIndependently(SGDClassifier(n_iter=30)))
#eval_per_customer_model(SVC(kernel='linear', C=0.001, max_iter=60), y_transformation=lambda y: y[:, 0])
#eval_per_customer_model(GradientBoostingClassifier(n_estimators=10), y_transformation=lambda y: y[:, 0])

#eval_per_customer_model(EachTaskIndependently(SVC(kernel='linear')))
#eval_per_customer_model(EachTaskIndependently(LinearSVC(C=0.01)))


#eval(EachTaskIndependently(SGDClassifier(n_iter=70, shuffle=True)))


#OneVsRestClassifier()
