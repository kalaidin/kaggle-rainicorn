#!/usr/bin/env python3
from sklearn.multiclass import OneVsRestClassifier

from per_customer_eval import *
from per_customer_models import *

eval_per_customer_model(LastQuoted)
#eval_per_customer_model(SGD, n_iter=30)

OneVsRestClassifier()
