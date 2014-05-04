#!/usr/bin/env python3
from matplotlib.pyplot import show, hist, plot
from commons import *

train = read_data('train.csv')
test = read_data('test.csv')

train.columns

variable = 'cost'
np.unique(train.drop_duplicates('customer_ID', take_last=True)[variable].values)

d = train.drop_duplicates('customer_ID', take_last=True)[variable].values
dn = d[~np.isnan(d)]
## change in cost?
## need a huge dataset where u can switch on and off between variables

np.log(car_age + 1).std()
hist(d, bins=30); show()

## 'car_value', 'risk_factor', 'age_oldest', 'age_youngest', 'married_couple', 'C_previous', 'duration_previous', 'cost'], dtype='object')
