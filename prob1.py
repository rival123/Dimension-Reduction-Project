# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:05:03 2018

@author: ASUS
"""
import math
import statistics
n = input('number of Rain Gauges: ')
err = input('Enter percent of error :')
err = float(err)
n = int(n)
set1 = []
for i in range(n):
    k = input('enter rainfall for rain gauge %d :' %(i+1))
    k = float(k)
    set1.append(k)
sigma = statistics.stdev(set1)
p_bar = statistics.mean(set1)
c_v = 100*sigma/p_bar
m = (c_v/err)**2
print('Number of stations Required :',m)
print('number of more stations required :',r)

    
    