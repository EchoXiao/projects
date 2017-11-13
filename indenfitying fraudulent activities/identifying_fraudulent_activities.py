'''
Goal
E-commerce websites often transact huge amounts of money. And whenever a
huge amount of money is moved, there is a high risk of users performing
fraudulent activities, e.g. using stolen credit cards, doing money laundry, et .

Machine Learning really excels at identifying fraudulent activities.
Any website where you put your credit card information has a risk team
in charge of avoiding frauds via machine learning.

The goal of this challenge is to build a machine learning model that
predicts the probability that the first transaction of a new user is fraudulent.

Challenge Description
Company XYZ is an e-commerce site that sells hand-made clothes.
You have to build a model that predicts whether a user has a high probability
of using the site to perform some illegal activity or not. This is a super
common task for data scientists.

You only have information about the user first transaction on the site and
based on that you have to make your classification ("fraud/no fraud").

These are the tasks you are asked to do:
1. For each user, determine her country based on the numeric IP address.
2. Build a model to predict whether an activity is fraudulent or not. Explain
how different assumptions about the cost of false positives vs false negatives
would impact the model.
3. Your boss is a bit worried about using a model she doesn't understand for
something as important as fraud detection. How would you explain her how the
model is making the predictions? Not from a mathematical perspective (she
couldn't care less about that), but from a user perspective. What kinds of
users are more likely to be classified as at risk? What are their characteristics?
4. Let's say you now have this model which can be used live to predict in
real time if an activity is fraudulent or not. From a product perspective,
how would you use it? That is,what kind of different user experiences would
you build based on the model output?
'''






import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# frauds = pd.read_csv('Fraud_Data.csv')
# ip = pd.read_csv('IpAddress_to_Country.csv')
# ip['upper_bound_ip_address'] = ip['upper_bound_ip_address'].apply(lambda x: float(x))


# for index in ip.index:
#    p = ip.iloc[index]
#    flt = (frauds['ip_address'] <= p['upper_bound_ip_address']) & (frauds['ip_address'] >= p['lower_bound_ip_address'])
#    frauds.loc[flt, 'country'] = p['country']
# frauds.to_csv('frauds_country.csv')
#
frauds_country = pd.read_csv("frauds_country.csv", index_col=False)
frauds_country.drop(frauds_country.columns[[0]], axis = 1)

print frauds_country.head()
frauds_country['country'] = frauds_country['country'].fillna('None')




frauds_country['purchased_days'] = pd.to_datetime(frauds_country['purchase_time']) - pd.to_datetime(frauds_country['signup_time'])
frauds_country['purchased_days'] = frauds_country['purchased_days'].apply(lambda x: x.days)



frauds_country = frauds_country.drop(['signup_time', 'purchase_time', 'ip_address'], axis = 1)



# X_train, X_test, y_train, y_test = train_test_split(frauds_country.drop(['class'], axis = 1), frauds_country['class'])
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# print dt.feature_importances_
