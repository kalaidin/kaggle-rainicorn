#!/usr/bin/env python3
from collections import defaultdict
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from functools import partial

from per_customer_eval import *
from per_customer_models import *

eval = partial(eval_per_customer_model, 'per-customer-1')
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

# same performance as LastQuoted 0.537753
#min_list = ['C_previous',
#            'last_A',
#            'last_B',
#            'last_C',
#            'last_D',
#            'last_E',
#            'last_F',
#            'last_G']
#eval(EachTaskIndependently(SGDClassifier(n_iter=5, shuffle=True)), include_variables=min_list)

l2 = ['viewed_A',
      'viewed_B',
      'viewed_C',
      'viewed_D',
      'viewed_E',
      'viewed_F',
      'viewed_G',
      'C_previous',
      'last_A',
      'last_B',
      'last_C',
      'last_D',
      'last_E',
      'last_F',
      'last_G',
      'order',
      'day_first_visit',
      'day_last_visit',
      'state',
      #'car_age',
      'car_age_norm',
      'car_value',
      'risk_factor',
      #'cost',
      'cost_norm',
      'age_oldest',
      'age_youngest',
      'age_difference',
      'age_prop'
      'age_youngest_ranged',
      'age_oldest_ranged',
      #'duration_previous',
      #'duration_previous_safe',
      'duration_previous_ranged',
      'married_couple',
      'group_size',
      'homeowner']

#estimator = EachTaskIndependently(SGDClassifier(loss='squared_hinge', n_iter=50, shuffle=True, n_jobs=-1, alpha=0.0004))
#make_per_customer_submission1('per-customer-1', estimator, include_variables=l2)
#make_per_customer_submission1('per-customer-1', estimator, include_variables=l_cont)

#eval(estimator, include_variables=l2)

l_cont = ['viewed_A',
          'viewed_B',
          'viewed_C',
          'viewed_D',
          'viewed_E',
          'viewed_F',
          'viewed_G',
          'C_previous',
          'last_A',
          'last_B',
          'last_C',
          'last_D',
          'last_E',
          'last_F',
          'last_G',
          'order',
          'day_first_visit',
          'day_last_visit',
          'state',
          #'car_age',
          'car_age_norm',
          'car_value',
          'risk_factor',
          #'cost',
          'cost_norm',
          'age_oldest',
          'age_youngest',
          'age_difference',
          'age_prop'
          'age_youngest_ranged',
          'age_oldest_ranged',
          #'duration_previous',
          #'duration_previous_safe',
          'duration_previous_ranged',
          'married_couple',
          'group_size',
          'homeowner']


def report(res):
    print('=' * 60)
    for k, ss in sorted(res.items(), key=itemgetter(1)):
        print('%s: %f (min=%f, max=%f, sd=%f)' % (k, np.mean(ss), np.min(ss), np.max(ss), np.std(ss)))
    print('-' * 60)
    print(dict(res))
    print('=' * 60)


#res = defaultdict(lambda *x: [])
#report(res)
#for n_estimators in [50, 200, 1000]:
#    for min_samples_split in [1000, 650, 400, 200]:
#        for cov_i, cov in enumerate(COVERAGE):
#            for i in range(3):
#                s = eval(RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, min_samples_split=min_samples_split),
#                    x_transformation=lambda x: x.todense(), include_variables=l4, y_transformation=lambda y: y[:, cov_i])
#                res[(cov, n_estimators, min_samples_split)].append(s)
#            report(res)
#report(res)

#eval(SGDClassifier(shuffle=True, loss='log', n_iter=50), y_transformation=lambda y: y[:, 0], include_variables=l_cont)
#eval(AsOrdinal(SGDClassifier(shuffle=True, loss='log', n_iter=100, n_jobs=-1, alpha=0.0004)),
#     y_transformation=lambda y: y[:, 2], include_variables=l2)


#eval(EachTaskIndependently(SGDClassifier(shuffle=True, n_iter=90, n_jobs=-1, alpha=0.0004)),
#         include_variables=l2, interacting_variables=create_interactions('last_X', order=3))

l5 = [
          'last_A',
          'last_B',
          'last_C',
          'last_D',
          'last_E',
          'last_F',
          'last_G',]


l4 = ['C_previous',
      'last_A',
      'last_B',
      'last_C',
      'last_D',
      'last_E',
      'last_F',
      'last_G',
      'viewed_A',
      'viewed_B',
      'viewed_C',
      'viewed_D',
      'viewed_E',
      'viewed_F',
      'viewed_G']

#for i, C in enumerate(COVERAGE):
#    eval(LastQuotedOnCategoricalLastX(),
#         x_transformation=lambda x: x.toarray(), include_variables=l5, y_transformation=lambda y: y[:, i])
#    print('for', C)
#    print('=' * 60)

eval(EachTaskIndependentlyWithRules(LastQuotedOnCategoricalLastX(),
                                    [('0011004', '0011002')]), include_variables=l5, x_transformation=lambda x: x.toarray())
#eval(JointEstimator([LastQuotedOnCategoricalLastX()] * 4), x_transformation=lambda x: x.toarray(), include_variables=l5,
#     wrapper_and_unwrapper=create_wrapper_and_unwrapper([0, 3], [1], [2, 4], [5, 6]))

#eval(SGDClassifier(shuffle=True, n_iter=50, n_jobs=-1, alpha=0.0004),
#     y_transformation=lambda y: y[:, 2], include_variables=l2)