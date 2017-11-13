#!/usr/bin/python 
# -*- coding: utf-8 -*-





'''
Goal
- Online shops often sell tons of different items and this can become very messy very quickly!
- Data science can be extremely useful to automatically organize the products in categories so that 
  they can be easily found by the customers.
- The goal of this challenge is to look at user purchase history and create categories of items 
  that are likely to be bought together and, therefore, should belong to the same section.

Challenge Description
- Company XYZ is an online grocery store. In the current version of the website, they have manually 
  grouped the items into a few categories based on their experience. However, they now have a lot 
  of data about user purchase history. Therefore, they would like to put the data into use!

This is what they asked you to do:
- The company founder wants to meet with some of the best customers to go through a focus group with 
  them. You are asked to send the ID of the following customers to the founder:
	1. the customer who bought the most items overall in her lifetime
 	2. for each item, the customer who bought that product the most
	3. Cluster items based on user co-purchase history. That is, create clusters of products
	   that have the highest probability of being bought together. The goal of this is to replace 
	   the old/manually created categories with these new ones. Each item can belong to just one cluster.
'''



import pandas as pd 
import numpy as np 
from collections import Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error



# Problems 3


item_to_id = pd.read_csv("item_to_id.csv")
item_to_id.columns = ['item_name', 'item_id']
purchase_history = pd.read_csv("purchase_history.csv")




# 1) Firstly have a table where rows are each transaction, 
# features are all possible items and value is 1/0 
# depending on whether it was bought or not within 
# that transaction. 

# user_item = purchase_history.drop('id', axis = 1).join(purchase_history.id.str.split(',', expand = True).stack().reset_index(drop = True, level = 1).rename("item_id"))
# user_item.item_id = user_item.item_id.apply(lambda x: int(x))
# user_dummies = pd.get_dummies(user_item.item_id).sum(level = 0)
# user_dummies.to_csv('user_dummies.csv')
user_dummies = pd.read_csv("user_dummies.csv")







# 2) Once you have this table, create the correlation 
# matrix of that table, so that you will have correlation 
# between features (here items).  So now you have a NxN 
# matrix where N is the number of items and each cell value 
# represents similarity between corresponding items.

corr = user_dummies.corr()



# 3) Apply k-means to this matrix. Your output will be the final clusters. 
inertia_list = []
n = 20


for k in range(1, n):
	print "=====================for %s clusters =================== " % k
	kmeans = KMeans(n_clusters = k, random_state = 1).fit(corr)
	kmeans_labels = pd.DataFrame(kmeans.labels_)
	kmeans_labels.index = range(1, len(kmeans_labels) + 1)
	kmeans_labels = kmeans_labels.reset_index(level = 0)
	kmeans_labels.columns = [['item_id', 'labels']]
	print pd.merge(kmeans_labels, item_to_id, how = 'inner', on = 'item_id').sort_values(by = 'labels')
	inertia_list.append(str(kmeans.inertia_))






x = [k for k in range(1, n)]
y = inertia_list
plt.plot(x, y)
plt.show()


for k in range(1, n):
	print "=====================for %s clusters =================== " % k
	ward = AgglomerativeClustering(n_clusters = k, linkage = 'ward').fit(corr)
	ward_labels = pd.DataFrame(ward.labels_)
	ward_labels.index = range(1, len(ward_labels) + 1)
	ward_labels = ward_labels.reset_index(level=0)
	ward_labels.columns = [['item_id', 'labels']]
	print pd.merge(ward_labels, item_to_id, how = 'inner', on = 'item_id').sort_values(by = 'labels')
















