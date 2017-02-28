#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns

salaries_df = pd.read_csv('baseball_stats/core/Salaries.csv')

# only get salaries from 2010
salaries_2010_df = salaries_df.drop(salaries_df[salaries_df['yearID']!=2010].index)

# sum salaries by team
sum_salaries_2010_df = salaries_2010_df.groupby('teamID', as_index=False).sum()

# get the team salary difference from the team salary mean
def diff_from_mean_column(column):
  return column - column.mean()

diff_from_mean_salaries = diff_from_mean_column(sum_salaries_2010_df['salary'])
sum_salaries_2010_df['salary'] = diff_from_mean_salaries

# plot team salaries
sns.set_style("whitegrid")
ax = sns.barplot(x='teamID',y='salary', data=sum_salaries_2010_df)
sns.plt.show()
