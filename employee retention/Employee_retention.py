# Goal
# Employee turn-over is a very costly problem for companies.
# The cost of replacing an employee if often larger than 100K USD,
# taking into account the time spent to interview and find a
# replacement, placement fees, sign-on bonuses and the loss of
# productivity for several months.
#  t is only natural then that data science has started being
#  applied to this area. Understanding why and when employees
#  are most likely to leave can lead to actions to improve
#  employee retention as well as planning new hiring in advance.
#  This application of DS is sometimes called people analytics
#  or people data science (if you see a job title: people data
#  scientist, this is your job).
#
# In this challenge, you have a data set with info about the
#  employees and have to 1. predict when employees are going to
#  quit by understanding the main drivers of eployee churn.
#
# Challenge Description
# We got employee data from a few companies. We have data
# about all employees who joined from 2011/01/24 to 2015/12/13.
# For each employee, we also know if they are still at the
# company as of 2015/12/13 or they have quit. Beside that,
# we have general info about the employee, such as avg salary
# during her tenure, dept, and yrs of experience.
# As said above, the goal is to predict employee retention and
# understand its main drivers. Specifically, you should:
# Assume, for each company, that the headcount starts from
# zero on 2011/01/23.

# 1. Estimate employee headcount, for each
# company, on each day, from 2011/01/24 to 2015/12/13.
# That is, if by 2012/03/02 2000 people have joined company
# 1 and 1000 of them have already quit, then company headcount
# on 2012/03/02 for company 1 would be 1000.
# 2. You should
# create a table with 3 columns: day, employee_headcount,
# company_id.
# 3. What are the main factors that drive employee churn? Do they
# make sense? Explain your findings.
# 4. If you could add to this data set just one variable that
# could help explain employee churn, what would that be?

import pandas as pd
import numpy as np
data = pd.read_csv("employee_retention_data.csv")
table = pd.read_csv("table.csv")


'''
join_cnt =  data.groupby(['company_id', 'join_date']).size().reset_index().rename(columns={0:"join_cnt"})
quit_cnt = data.groupby(['company_id', 'quit_date']).size().reset_index().rename(columns={0:"quit_cnt"})

table = pd.merge(join_cnt, quit_cnt, left_on = ['company_id', 'join_date'], right_on = ['company_id', 'quit_date'])
table['difference'] = table['join_cnt'] - table['quit_cnt']
table = table[['company_id', 'join_date', 'difference']].rename(columns = {'join_date': 'day'})
table['headcounts'] = table.groupby(['company_id']).cumsum()
table.to_csv("table.csv", index = False)
'''







print data[(data['company_id'] == 1) & (data['join_date'] == '2012-01-03')]














