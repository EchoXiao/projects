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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import mean_squared_error


item_to_id = pd.read_csv("item_to_id.csv")
purchase_history = pd.read_csv("purchase_history.csv")


#########Comments##############
#There are 2 problems here:
# 1) You are picking the line with the max purchases. However, the same user might have
# multiple purchases. Like take user_id 223. She has two transactions and overall she bought 11 items (see below).
# So after for each user you estimate how many items they bought in a single transaction, you need to group by user_id
# and sum to find the total. 

# 223                  1,6,35,29,41,45,32   
# 223                          17,2,22,38   

# 2) When you calculate user_id_max, you are actually returning the index, not the user_id. Something like below would give you the user_id
# purchase_history['user_id'][purchase_history['id'].apply(lambda x: len(set(x.split(',')))) == max_cnt]    
#########End Comments##############

# Solution1
max_cnt = purchase_history['id'].apply(lambda x: len(set(x.split(',')))).max()
[idx, user_id_max] = purchase_history['id'][purchase_history['id'].apply(lambda x: len(set(x.split(',')))) == max_cnt].index.tolist()
print user_id_max
# Customer 16424 bought the most items.





# ##########Comments###########
# This works, but here the question was different. For each item, you should find the customer
# who bought it the most. So like, Sugar was bought the most by customer X, lettuce by customer y, etc.
# #############################
# Solution2
concat_list = []
count_list = []
purchase_history['id'].apply(lambda x: count_list.extend(x))
c = Counter(count_list)
purchase_history['id'].apply(lambda x: concat_list.append(x.split(',')))
concated_list = list(itertools.chain.from_iterable(concat_list))
c = Counter(concated_list)
item_purchased_most = item_to_id[item_to_id['Item_id'] == int(max(c, key = c.get))]['Item_name'].iloc[0]
print 'The most purchased item is: ' + item_purchased_most





# Solution3
purchase_history['id'] = purchase_history.id.apply(lambda x: x.split(','))
item_id = purchase_history['id'].apply(pd.Series).stack().reset_index(level = 1, drop = True).to_frame()
item_user = pd.merge(purchase_history, item_id, left_index = True, right_index = True).drop('id', axis = 1)
item_user.columns = (['user_id', 'item_id'])
user_id = item_user.groupby('item_id', as_index = False)[['user_id']].agg(lambda x: list(x))
item_purchased = user_id.user_id.apply(lambda x: ",".join(str(item) for item in x)).to_frame()
item_history = pd.merge(item_purchased, user_id, left_index = True, right_index = True).drop('user_id_y', axis = 1)
item_history.to_csv("item_history.csv", sep = ',', index=False)


print item_id.head()




item_history = pd.read_csv("item_history.csv", sep = ',')
item_history.columns = ['user_id', 'item_id']
vec = CountVectorizer()
X = vec.fit_transform(item_history['user_id'])
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(X)
labels = pd.DataFrame(kmeans.labels_)
labels.columns = ['groups']
item_groups = pd.merge(labels, item_history, left_index = True, right_index = True)
item_groups.to_csv('item_groups.csv', index = False)
item_groups = pd.read_csv("item_groups.csv", sep = ",")

item_groups[item_groups['groups'] == 0].to_csv("group_0.csv", index = False)
item_groups[item_groups['groups'] == 1].to_csv("group_1.csv", index = False)
item_groups[item_groups['groups'] == 2].to_csv("group_2.csv", index = False)




# n = [3, 4, 5, 6, 7]
# for i in n:
#       kmeans = KMeans(n_clusters = i, random_state = 0).fit(X)
#       print kmeans.score(X)
#       -237097.016667
#       -222547.970588
#       -216170.34902
#       -206600.351075
#       -197739.665591
















1. How can I measure the performance of k-means algorithm? How do I know if it is the appropriate algorithm? Plotting is a very direct method in getting the best algorithm. What can I do with multi dimensional data? What about this one?

# Clustering is unsupervised ML, so you don't know the actual 
# labels of each data point. Matter of fact, they don't exist. 
# Therefore, there is no such thing as 1 number that 
# tells you exactly how well your model is 
# doing. A common way to estimate clustering performance 
# is looking at sum of squared errors, like 
# score does in python. That gives you an idea of 
# whether points in your clusters are similar 
# (good) or they are very spread out (bad) and it tells 
# you when there is no point in keeping adding new 
# clusters. So it can be used to optimize the number of clusters. 
# That being said, in practice, after you optimize the number 
# of clusters, you will simply manually check your clusters. 
# Take a bunch of data points and see to which clusters they get 
# assigned. See if it makes sense or not. 

The most common step by step approach to clustering is:

# 1) Try kmeans and optimize number of clusters
# 2) Check the results and see if they make sense
# 3) If not, move to hierarchical clustering. 
# Hierarchical clustering is more powerful than kmeans, 
# but more complicated. So you tend to use it only 
# if kmeans fails. In general, kmeans does badly 
# in high dimensions. So if you have a lot of 
# variables, expect hierarchical clustering to do much better. 


# 2. I just randomly choose the number of centers because the score grows larger as clusters increases. So how I can pick the best number of centers in this question?

# The closer to zero, the better it is. Python shows the opposite of squared distance, meaning large negative number -> bad, close to zero -> good. So your score is improving with more clusters. Plot the score for each cluster number. You will see that the score keeps improving, until a point where it flattens and the more clusters you add, it doesn't really improve much. That's your optimal number of clusters. 
# 3. Should I use other  clustering algorithms and make assembler? I think it is not necessary to use assembler and more complicated algorithms since this is a very small, simple dataset. 

# Here kmeans is perfect cause it is simple. Always in DS, if you can use the simplest method, use that. Only move to more advanced stuff if the simplest technique doesn't work well.
# 4. I feel like I have to do more coding. But I do not know which is the most important thing I should do for now. What should I focus when I am trying to fix these problems? 

# If you just want to improve your coding skills, do some challenges from Euler project. Especially the simplest are very useful for DS cause the majority of DS coding is about data manipulation. And those projects will help you with that. 
# 5. I tried to do feature engineering. one character called 'how many times does the item was being bought' would be a good feature I thought. Is there any other ways to get more features? 

# Not here. This was a straightforward clustering problem based on co-occurrences. In pretty much all other challenges, you will have to do feature engineering.
# 6. What should I do to get better clustering and classifying groups? All I did is trying to find good features and apply sklearn machine learning algorithms simply. What else can I do to get results more accurate, especially in real world situation. 

# Building a model is just a small part of the work. Then you need to look at the results and refine your model. Like here, look at the clusters and see if they make sense. Can you give a name/label to each cluster? I.e. similar items are ending up in the same cluster? Also, how would you deal with items that end up belonging to a given cluster but are very far from the cluster centroid? For instance, you could think about creating a specific cluster called "other" for all items that don't quite belong to any cluster. The most important thing is: imagine you were actually working. This company is asking you to give them groups of items that get bought together so they can create categories. That requires some post-processing and making sure final results are sound from both a DS and a product standpoint. 
# Similarly if you do supervised ML, after the classification, there is a lot of work involved into looking at false positive vs false negative cost, choosing the right cut-off from a product standpoint and giving final results.  
 
# 7. Is there any website or books with high quality solutions to specific machine learning problems? I found some kaggle solutions in github, but I found it is hard to learn. I feel reading good problem solving codes would help me learn better.

# The 4 solutions in the book should help you with that. That kind of step by step approach can be used pretty much always. You picked a very specific problem about clustering which is different from all the others. But if you do the common ML problems, those solutions should help you. Anything that's not clear there? 

# I am attaching a file with comments on your code for the first 2 questions. For the actual clustering exercise, you want the score to go down. So play with the cluster number and find the best one. Then show what are the actual clusters and try to give a name to each cluster. Eventually similar items should end up in the same cluster, having at most one cluster called "other" where you have items not related to each other.  


