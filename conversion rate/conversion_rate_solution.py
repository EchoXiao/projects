import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt

data = pd.read_csv("conversion_data.csv")
data.drop([90928,295581], axis = 0).sort_values(by = 'age', ascending = False).head(7)


df_analysis = data.groupby(['source', 'country'])[['converted']].count()


X = data[data.columns.values[:-1]]
y = data['converted']

lb = LabelEncoder()
X['country'] = lb.fit_transform(X['country'])
X['source'] = lb.fit_transform(X['source'])

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline  = Pipeline(steps = [('clf', DecisionTreeClassifier(criterion = 'entropy'))])
parameters = [{
    'clf__max_depth': (150, 155, 160),
    'clf__min_samples_split': (1, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}]



grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, error_score = 0)
grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_estimator_.get_params()


# Predicting conversion rate
preds = grid_search.predict(X_test)
classification_report(y_test, preds)



# Using Ensemble Methods
pipeline = Pipeline(steps = [('clf', RandomForestClassifier(criterion='entropy'))])
clf_forest = RandomForestClassifier(n_estimators = 20, criterion='entropy',
                                    max_depth = 50, min_samples_leaf = 3,
                                    min_samples_split = 3, oob_score = True)
clf_forest.fit(X_train, y_train)
preds = clf_forest.predict(X_test)
classification_report(y_test, preds)
clf_forest.feature_importances_


clf_forest.oob_score_
# Out of box score is 98.5%. This implies, the predictions were right
# for 98.5% of the data set. However, it is useful to recognize that
# the dataset is highly imbalanced with 97% of the data in class 0 and
# only 3% in class 1. This implies, even if we had classified everything
# as 0, the accuracy would still be 97%. Therefore, our predictions have
# only improved by 1.5% which is the difference of 98.5% and 97%.
#
# The real challenge lies in improving the recall value of class 1.
# This value is close to 70%. This implies 30% of the customers who
# converted were not recognized by our system as potential customers.
#
# Recall value needs to be improved even if that results in an increased
# overall error rate (decrease in precision) and a decrease in specificity value.



# Building Random Forest Again: without the total pages visited feature.
# Since the classes are heavily imbalanced, giving it weights.



X_new_train = X_train.copy()
X_new_test = X_test.copy()
X_new_train = X_new_train.drop('total_pages_visited', axis = 1)
X_new_test = X_new_test.drop('total_pages_visited', axis = 1)

clf_forest_new = RandomForestClassifier(n_estimators = 20,
                                        criterion = 'entropy',
                                        max_depth = 150,
                                        min_samples_leaf = 2,
                                        min_samples_split = 2,
                                        oob_score = True,
                                        class_weight = {0: 0.1, 1:0.9})
clf_forest_new.fit(X_new_train, y_train)
preds = clf_forest_new.predict(X_new_test)
classification_report(y_test, preds)

clf_forest_new.feature_importances_
# recall value has significantly crashed.






#
# Conclusions
#
# It is possible to predict upto 98.5% accuracy or an out of
# bag error rate of 1.5%.
#
# Some of the important features are
#
# 1. Number of pages visited
# 2. new_user
# 3. country
# 4. age
#
# Site is working very well for Germany and not so well for China.
#
# This could be because of various reasons such as poor translation or the
# chineese site might be in English as well.
#
# The site works well for younger people. Those less than 30 years of age.
# It could either be because of the nature of the site that caters only
# to young people or it could be possible that older people find it
# difficult to navigate through the site for a meaningful conversion.

# Also, since the most important feature here is the number of pages
# visited, it implies that, if someone who has visited a lot of pages
# hasn't converted, then there is a good chance that the user will convert '
#     'with a little bit of marketing such as discounts, benefits etc.

# Also, users with an old account tend to do better than the ones with
# a new account. This could be used by marketing team to it's advantage.





