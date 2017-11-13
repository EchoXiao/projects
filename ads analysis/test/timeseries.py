from statsmodels.graphics.tsaplots import plot_acf
from pandas import Series
from matplotlib import pyplot
# load dataset
series = Series.from_csv('car-sales.csv', header=0)
# seasonal difference
differenced = series.diff(12)
# trim off the first year of empty data
differenced = differenced[12:]
# save differenced dataset to file
differenced.to_csv('seasonally_adjusted.csv')
# plot differenced dataset

seies = Series.from_csv("seasonally_adjusted.csv", header = None, sep = ",")
print series
plot_acf(series)
pyplot.show()





