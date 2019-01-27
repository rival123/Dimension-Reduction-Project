# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 17:10:24 2018

@author: Rahul
"""

import pandas as pd
import numpy as np
energy = pd.read_excel('Energy Indicators.xls')
energy = energy[16:243]
energy = energy.drop(energy.columns[[0,1]], axis = 1)
energy.rename(columns = {'Environmental Indicators: Energy':'Country','Unnamed: 3':'Energy Supply','Unnamed: 4':'Energy Supply per Capita','Unnamed: 5':'% Renewable'}, inplace=True)
energy.replace('...', np.nan,inplace = True)
energy['Energy Supply'] = energy['Energy Supply']*1000000
for s in range(16,242):
    result = ''.join([i for i in energy['Country'][s] if not i.isdigit()])
    i = result.find('(')
    if i>-1: result = result[:i]
    result = result.strip()
    energy['Country'].replace(energy['Country'][s],result,inplace = True)
di = {"Republic of Korea": "South Korea","United States of America": "United States","United Kingdom of Great Britain and Northern Ireland": "United Kingdom","China, Hong Kong Special Administrative Region": "Hong Kong"}
energy.replace({'Country':di},inplace = True)
GDP = pd.read_csv('world_bank.csv', skiprows = 4)
GDP.rename(columns = {'Country Name': 'Country'}, inplace = True)
di = {"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran","Hong Kong SAR, China": "Hong Kong"}
GDP.replace({'Country':di},inplace = True)
ScimEn = pd.read_excel('scimagojr-3.xlsx')
df = pd.merge(pd.merge(energy,GDP,on='Country'),ScimEn,on = 'Country')
df.set_index('Country',inplace=True)
df = df[['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']]
df = (df.loc[df['Rank'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])])
df.sort_values('Rank',inplace = True)
def answer_one():
    return df
answer_one()
