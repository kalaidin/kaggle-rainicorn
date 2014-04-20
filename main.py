#!/usr/bin/env python3
from sklearn.svm import LinearSVC, SVC
import sys

from per_customer_eval import *
from per_customer_models import *

#A: 0.875
#B: 0.885
#C: 0.868
#D: 0.901
#E: 0.887
#F: 0.871
#G: 0.800


#eval_per_customer_model(LastQuoted)
#eval_per_customer_model(EachTaskIndependently(SGDClassifier(n_iter=30)))
eval_per_customer_model(SVC(kernel='linear', C=0.001), y_transformation=lambda y: y[:, 0])

#eval_per_customer_model(EachTaskIndependently(SVC(kernel='linear')))
#eval_per_customer_model(EachTaskIndependently(LinearSVC(C=0.01)))


#OneVsRestClassifier()
