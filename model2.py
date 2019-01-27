# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:13:01 2018

@author: Rahul
"""
import pandas as pd
census_df = pd.read_csv('census.csv')
census_df.head()
def answer_six():
    sumlev = census_df.SUMLEV.values == 50
    data = census_df[['CENSUS2010POP', 'STNAME', 'CTYNAME']].values[sumlev]
    s = pd.Series(data[:, 0], [data[:, 1], data[:, 2]], dtype=np.int64)
    def sum_largest(x, n=3):
        return x.nlargest(n).sum()
    return s.groupby(level=0).apply(sum_largest).nlargest(3).index.tolist()
answer_six()
#df = census_df[census_df['SUMLEV'] ==50]
#st = census_df.STNAME.unique()
#df1 = df.set_index(['STNAME', 'CTYNAME'])
#sum1 = []
#for name in st:
 #   df2 = df1.loc[name]['CENSUS2010POP']
  #  copydf2 = df2.copy()
   # sorted1 = copydf2.sort_values(ascending = False)
    #temp = sorted1[0] + sorted1[1] + sorted1[2]
    #sum1.append(temp)
#df2 = df1.loc['Alabama']['CENSUS2010POP']
#copydf2 = df2.copy()
#sorted1 = copydf2.sort_values(ascending = False)
#sorted1[0]
#st
