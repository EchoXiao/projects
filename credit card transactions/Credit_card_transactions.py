
# Goal
# One of the greatest challenges in fraud, and in general in that area of data science related to catching illegal activities, is that you often find yourself one step behind.
# 1. Your model is trained on past data. If users come up with a totally new way to commit a fraud, it often takes you some time to be able to react. By the time you get data about that new fraud strategy and retrain the model, many frauds have been already committed.
# 2. A way to overcome this is to use unsupervised machine learning, instead of supervised. With this approach, you don't need to have examples of certain fraud patterns in order to make a prediction. Often, this works by looking at the data and identify sudden clusters of unusual activities.
# 3. This is the goal of this challenge. You have a dataset of credit card transactions and you have to identify unusual/weird events that have a high chance of being a fraud.

# Challenge Description
# Company XYZ is a major credit card company. It has information about all the transactions that users make with their credit card.
# Your boss asks you to do the following:
# 1. Your boss wants to identify those users that in your dataset never went above the monthly credit card limit month (calendar month). The goal of this is to automatically increase their limit. Can you send him the list of Ids?
# 2. On the other hand, she wants you to implement an algorithm that as soon as a user goes above her monthly limit, it triggers an alert so that the user can be notified about that. We assume here that at the beginning  of  the new  month, user total money spend get reset to  zero  (i.e. she pays the card fully at the end of the month). Build a function that for each day, returns a list of users who went above their credit card monthly limit on that day.
# 3. Finally, your boss is very concerned about frauds cause they are a huge cost for credit card companies. She wants you to implement an unsupervised algorithm that returns all transactions that seem unusual and are worth being investigated further.



import pandas as pd 
from datetime import datetime

cc_info = pd.read_csv("cc_info.csv")
transactions = pd.read_csv("transactions.csv")
# transactions['date'].apply(lambda x: x)

print datetime.strptime("transactions['date'][1]", "%Y%m%d")


# # print cc_info.head()
# # print transactions.head()


# # ===========Solution 1===========================================

# tran_sum = transactions.groupby(["credit_card"], as_index = False).sum()[['credit_card', 'transaction_dollar_amount']]
# cc_tran_lim = pd.merge(cc_info, tran_sum, how = "inner", on = "credit_card")
# good = cc_tran_lim[cc_tran_lim["transaction_dollar_amount"] < cc_tran_lim["credit_card_limit"]]["credit_card"]

# print ("There are %s credit card owner never pass the limit. \n" % len(good))
# # print ("The list of the credit card owner are %s" % good.values)


# # ===========Solution 2===========================================


# print transactions.head()