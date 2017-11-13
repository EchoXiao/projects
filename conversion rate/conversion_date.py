
# Goal
# Optimizing conversion rate is likely the most common work of a data scientist,
# and rightfully so.
# The data revolution has a lot to do with the fact that now we are able to
# collect all sorts of data about people who buy something on our site as well
# as people who don't. This gives us a tremendous opportunity to understand what's
# working well (and potentially scale it even further) and what's not working
# well (and fix it).


# The goal of this challenge is to
# 1. build a model that predicts conversion rate and, based on the model,
# 2. come up with ideas to improve revenue.

# This challenge is significantly easier than all others in this collection.
# There are no dates, no tables to join, no feature engineering required,
# and the problem is really straightforward. Therefore, it is a great starting
# point to get familiar with data science take home challenges.
# You should not move to the other challenges until you fully understand this one.

# Challenge Description
# We have data about users who hit our site: whether they converted
# or not as well as some of their characteristics such as their country,
# the marketing channel, their age, whether they are repeat users and the
# number of pages visited during that session (as a proxy for site activity/time
# spent on site).

# Your project is to:
# 1. Predict conversion rate
# 2. Come up with recommendations for the product team and the marketing
# team to improve conversion rate

# Conclusion:
 # Their country, source is not important to the final conversion step. But
 #  features like 'age', 'total pages visited' and 'new_user' have significant
 #  effect towards conversion.



from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def preprocessed(loc):

    data = pd.read_csv(loc)
    data['intercept'] = 1

    age_1 = 25
    age_2 = 50

    country_dummied = pd.get_dummies(data['country'], drop_first=True)
    user_dummied = pd.get_dummies(data['new_user'], prefix="user", drop_first=True)
    source_dummied = pd.get_dummies(data['source'], drop_first=True)

    age_dummies = pd.get_dummies(pd.cut(data['age'], bins=[data['age'].min(), age_1, age_2, data['age'].max()]),
                                 drop_first=True)
    scaler = MinMaxScaler()
    scaler.fit(data['total_pages_visited'].values.reshape(-1, 1))
    data['visited_scaled'] = pd.DataFrame(scaler.transform(data['total_pages_visited'].values.reshape(-1, 1)))



    df = pd.concat([country_dummied,
                    user_dummied,
                    source_dummied,
                    data[['age', 'total_pages_visited', 'intercept', 'converted']]], axis=1)

    # df = pd.concat([country_dummied,
    #                 user_dummied,
    #                 source_dummied,
    #                 age_dummies,
    #                 data[['visited_scaled', 'intercept', 'converted']]], axis = 1)


    return df

# random forest
def sk_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier().fit(X_train, y_train)
    rf.feature_importances_
    y_true, y_pred = y_test, rf.predict(X_test)
    (classification_report(y_true, y_pred))

def sk_dtree(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier().fit(X_train, y_train)
    dt.feature_importances_
    dt.score
    y_true, y_pred = y_test, dt.predict(X_test)
    (classification_report(y_true, y_pred))



# Sklearn
def sk_logit(X_train, y_train, X_test, y_test):

    lr = LogisticRegression().fit(X_train, y_train)
    lr.score(X_train, y_train)

    y_true, y_pred = y_test, lr.predict(X_test)
    (classification_report(y_true, y_pred))
    lr.coef_




# SM
def sm_logit(y_trian, X_train, X_test):
    logit = sm.Logit(y_train, X_train).fit()
    y_pred = logit.predict(X_test)
    logit.summary()
    np.exp(logit.params)

    params = logit.params
    conf = logit.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    np.exp(conf)


def decision_tree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier().fit(X_train, y_train)

    y_true, y_pred = y_test, clf.predict_proba(X_test)




if __name__ == "__main__":
    loc = "conversion_data.csv"
    df = preprocessed(loc)

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1],
                                                        df.iloc[:, -1],
                                                        test_size=0.3,
                                                        random_state=42)
    sk_dtree(X_train, y_train, X_test, y_test)
    sk_rf(X_train, y_train, X_test, y_test)
    sk_logit(X_train, y_train, X_test, y_test)
    sm_logit(y_train, X_train, X_test)
    decision_tree(X_train, y_train, X_test, y_test)
