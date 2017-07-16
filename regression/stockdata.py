import math
import quandl

import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression



quandl.ApiConfig.api_key = 'JkDjXva81_ujYvG2Swcr'

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

# Prediction price for next ten days - 0.1
forecast_out = int(math.ceil(0.01*len(df)))

# Shift the label col above by forecast_out
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

'''
Date        |  Adj. Close  HL_PCT    PCT_change  Adj. Volume  |    label
-----------------------------------------------------------------------------
2004-08-19  |  50.322842  8.072956   0.324968   44659000.0    |   68.752232
2004-08-20  |  54.322689  7.921706   7.227007   22834300.0    |   69.639972
2004-08-23  |  54.869377  4.049360  -1.227880   18256100.0    |   69.078238
2004-08-24  |  52.597363  7.657099  -5.726357   15247300.0    |   67.839414
2004-08-25  |  53.164113  3.886792   1.183658    9188600.0    |   68.912727

'''

#
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

# Scaling out features in range of -1 to 1, speed up process and improves the accuracy.
X = preprocessing.scale(X)


#  Training and Testing
# Shuffle and Spliting dataset into training - 80 and testing - 20 = test_size = 0.2
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Using sklearn classifier - LinearRegression
print("Linear Regression:")
clf = LinearRegression(n_jobs=-1) # -1 run jobs as many as possible
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print("linear: {}".format(accuracy))
print("--------------------------------")

print("Support Vector:")
# Comparing kernal in SVR
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
  clf = svm.SVR(kernel=k)
  clf.fit(X_train, y_train)
  accuracy = clf.score(X_test, y_test)
  print(k, accuracy)


