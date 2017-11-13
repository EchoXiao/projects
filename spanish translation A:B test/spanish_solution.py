import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import scipy as sc
from scipy import stats
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from StringIO import StringIO
from inspect import getmembers



test = pd.read_csv("test_table.csv")
user = pd.read_csv("user_table.csv")
data = pd.merge(test, user, on = 'user_id', how = 'outer')
data['ads_channel'] = data['ads_channel'].fillna(0)



data_new = data.copy()
data_new = data_new[data_new['country'] != 'Spain']

zero = data_new[data_new['test'] == 0]
one = data_new[data_new['test'] == 1]

# print sc.stats.ttest_ind(zero['conversion'], one['conversion'], equal_var=False, axis = 0)

data_new['date'] = https://weibo.com/1879778060/Ft10ulYsS?ref=feedsdk&type=comment#_rnd1509605522707.to_datetime(data_new['date'], infer_datetime_format=True)
time_series = data_new.groupby(['date', 'test'])[['conversion']].mean()
time_series = time_series.unstack()['conversion'][1]/time_series.unstack()['conversion'][0]

fig, ax = plt.subplots(1, 1)
ax.plot(time_series)
ax.set_xlabel('Date')
ax.set_ylabel('test/control')
ax.set_title('Line Plot')
ax.xaxis.set_major_locator(ticker.MultipleLocator())

X = data_new.copy()
y = data_new['test']


lb = LabelEncoder()

# lb_cols = ['source', 'country', 'device', 'browser_language', 'ads_channel', 'browser', 'sex']
# for c in lb_cols:
#     X[c] = lb.fit_transform(X[c])
# X = X.drop(['conversion', 'date', 'test'], axis = 1)
#
#
#
# index = X['age'].index[X['age'].apply(np.isnan)]
# impute = Imputer(missing_values='NaN', strategy='median', axis = 0, copy = True)
# X['age'] = impute.fit_transform(X['age'].reshape(-1, 1))
#
#
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=2, min_samples_split=2)
# clf.fit(X, y)
# clf.feature_importances_


# zip(X.columns[clf.tree_.feature], clf.tree_.threshold, clf.tree_.children_left, clf.tree_.children_right)












