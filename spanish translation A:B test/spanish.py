'''

Goal
A/B tests play a huge role in website optimization.
Analyzing A/B tests data is a very important data scientist
responsibility. Especially, data scientists have to make sure
that results are reliable, trustworthy, and conclusions can be drawn.

Furthermore, companies often run tens, if not hundreds, of A/B tests
at the same time. Manually analyzing all of them would require lot of
time and people. Therefore, it is common practice to look at the
typical A/B test analysis steps and try to automate as much as
possible. This frees up time for the data scientists to work
on more high level topics.

In this challenge, you will have to analyze results from
an A/B test. Also, you will be asked to design an algorithm to
automate some steps.

Challenge Description
Company XYZ is a worldwide e-commerce site with localized versions of the site.
A data scientist at XYZ noticed that Spain-based users have a
much higher conversion rate than any other Spanish-speaking country.
She therefore went and talked to the international team in charge of
 Spain And LatAm to see if they had any ideas about why that was happening.

Spain and LatAm country manager suggested that one reason could be
translation. All Spanish- speaking countries had the same translation of
the site which was written by a Spaniard. They agreed to try a test
where each country would have its one translation written by a local.
That is, Argentinian users would see a translation written by an Argentinian,
Mexican users by a Mexican and so on. Obviously, nothing would change
for users from Spain.

After they run the test however, they are really surprised cause
the test is negative. I.e., it appears that the non-localized translation
was doing better!
You are asked to:

1. Confirm that the test is actually negative. That is, it appears
that the old version of the site with just one translation across Spain and
LatAm performs better
2. Explain why that might be happening. Are the localized translations really worse?
3. If you identified what was wrong, design an algorithm that would return
FALSE if the same problem is happening in the future and TRUE if everything
is good and the results can be trusted.
'''



import pandas as pd
from decimal import *
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier



user = pd.read_csv("user_table.csv")
test = pd.read_csv("test_table.csv")

data = pd.merge(test, user, how = 'inner', on = 'user_id')
data['ads_channel'] = data['ads_channel'].fillna(0)

test = data[data['test'] == 1]
control = data[data['test']== 0]
control_Span = control[control['country'] == 'Spain']
control_notSpan = control[control['country'] != 'Spain']

test['conversion'].mean()
control_Span['conversion'].mean()
control_notSpan['conversion'].mean()

# test(localized nonspanish) 4.3%
# control (localized spanish, nonlocalized nonspanish) 7.97%, 4.83%

# Spanish has better conversion. The localized language doesn't
# affect the conversion. Therefore, we have to compare the difference
# between localized Spanish data and localized non Spanish data.


localized = pd.concat([test, control_Span])
localized['spain'] = np.where(localized['country'] == 'Spain', 1, 0)
zero = localized[localized['spain'] == 0]
one = localized[localized['spain'] == 1]


from matplotlib import DataFrame, Series
from matplotlib.ticker as ticker
import scipy as sc
from scipy import stats
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
from sklearn.cross_validation import

sc.stats.ttest_ind()




