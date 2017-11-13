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
from collections import Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error





item_to_id = pd.read_csv("item_to_id.csv")
item_to_id.columns = ['item_name', 'item_id']
item_to_id['item_id'] = item_to_id['item_id'].apply(lambda x: str(x))
purchase_history = pd.read_csv("purchase_history.csv")


# Problem 1

user_item = purchase_history.drop('id', axis = 1).join(purchase_history.id.str.split(',', expand = True).stack().reset_index(drop = True, level = 1).rename("item_id"))
user_cnt = user_item.groupby('user_id', as_index = False).count()
# print ("User %s bought the most items in his/her overall lifetime.\n" % user_cnt.loc[user_cnt['item_id'].idxmax()]['user_id'])


# Problem 2
user_item_cnt = user_item.groupby(['item_id', 'user_id']).size().reset_index()
user_item_cnt.columns = ['item_id', 'user_id', 'cnt']
user_item_cnt.loc[user_item_cnt.groupby('item_id')['cnt'].idxmax()].to_csv('item_user_most.csv', index = False)



# Problem 3

item_history = user_item.groupby('item_id')['user_id'].agg(lambda x: list(x)).reset_index()
item_history.columns = ['item_id', 'user_id']
item_history['user_id'] = item_history['user_id'].apply(lambda x: ",".join(str(item) for item in x)).to_frame()



vec = CountVectorizer()
X = vec.fit_transform(item_history['user_id'])




m = 20
s = 3
interia_list = []
for n in range(s, m+1):

	#kmeans
	print "========== kmeans clustering =================\n"
	kmeans = KMeans(n_clusters = n, random_state = 1)
	kmeans.fit(X)
	item_labels = pd.merge(pd.DataFrame(kmeans.labels_), item_to_id, right_index = True, left_index = True)
	item_labels.columns = ['labels', 'item_item', 'item_id']
	interia = kmeans.inertia_
	interia_list.append(interia)
	for km_k in range(1, n+1):
		km_cluster_k = item_labels[item_labels['labels'] == km_k - 1]
		print km_cluster_k



	# hierarchy clustering
	print "========== hierarchy clustering =================\n"
	ward = AgglomerativeClustering(n_clusters= n, linkage = 'ward')
	ward.fit(X.toarray())
	item_labels = pd.merge(pd.DataFrame(ward.labels_), item_to_id, right_index = True, left_index = True)
	item_labels.columns = ['labels', 'item_name', 'item_id']
	for hc_k in range(1, n+1):
		hc_cluster_k = item_labels[item_labels['labels'] == hc_k-1]
		print hc_cluster_k



x = [n for n in range(s, m+1)]
y = interia_list
plt.plot(x, y)
plt.show()


# This plot did not show any turning point which indicates kmeans algorithm is not a good algorithm for this project.
# Divide the clusters into 8 groups looks good in hierarchy clustering.