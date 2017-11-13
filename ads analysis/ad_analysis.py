Goal
Maybe the first industry to heavily rely on data science was the online 
ads industry. Data Science is used to choose which ads to show, how 
much to pay, optimize the ad text and the position as well as in 
countless of other related applications.

Optimizing ads is one of the most intellectually challenging jobs a 
data scientist can do. It is a really complex problem given the huge 
(really really huge) size of the datasets as well as number of features 
that can be used. Moreover, companies often spend huge amounts of money 
in ads and a small ad optimization improvement can be worth millions 
of dollars for the company.

The goal of this project is to look at a few ad campaigns and analyze 
their current performance as well as predict their future performance.


Challenge Description
1. Company XYZ is a food delivery company. Like pretty much any other 
site, in order to get customers, they have been relying significantly 
on online ads, such as those you see on Google or Facebook.
2. At the moment, they are running 40 different ad campaigns and want 
you to help them understand their performance.

Specifically, you are asked to:
1. If you had to identify the 5 best ad groups, which ones would they 
be? Which metric did you choose to identify the best ad groups? Why? 
Explain the pros of your metric as well as the possible cons.
2. For each group, predict how many ads will be shown on Dec, 15 
(assume each ad group keeps following its trend).
3. Cluster ads into 3 groups: the ones whose avg_cost_per_click is 
going up, the ones whose avg_cost_per_click is flat and the ones whose 
avg_cost_per_click is going down.


>>> ad.head()
         date  shown  clicked  converted  avg_cost_per_click  total_revenue  \
0  2015-10-01  65877     2339         43                0.90         641.62   
1  2015-10-02  65100     2498         38                0.94         756.37   
2  2015-10-03  70658     2313         49                0.86         970.90   
3  2015-10-04  69809     2833         51                1.01         907.39   
4  2015-10-05  68186     2696         41                1.00         879.45   

           ad     cost  
0  ad_group_1  2105.10  
1  ad_group_1  2348.12  
2  ad_group_1  1989.18  
3  ad_group_1  2861.33  
4  ad_group_1  2696.00  


2. For each group, predict how many ads will be shown on Dec, 15 
(assume each ad group keeps following its trend).



import pandas as pd 



ad = pd.read_csv("ad_table.csv")
ad['cost'] = ad['avg_cost_per_click'] * ad['clicked']
ad_groups = ad[['shown', 'clicked', 'converted', 'cost', 'total_revenue', 'ad']].groupby('ad').sum()
ad_groups['cpa'] = (ad_groups['cost']) / ad_groups['converted']
print [n for n in ad_groups.sort_values(by = 'cpa').head(5).index]