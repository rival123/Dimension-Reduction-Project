# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:19:20 2018

@author: ASUS
"""
import numpy as np
import math
import statistics
from numpy import linalg as LA
n = input('number of stations: ')
n = int(n)
coordi= np.empty(shape = [n,2])
coordi_u= np.empty(shape = [1,2])
n_ann_ppt= np.empty(shape = [n])
ppt= np.empty(shape = [n])
d = np.empty(shape = [n])
r_x1 = 0
r1=0
r2=0
coordi_u[0][0] = float(input('x coordinate of unknown ppt station :'))
coordi_u[0][1] = float(input('y coordinate of unknown ppt station :'))
n_ann_ppt_u = float(input('Enter normal annual precipitation of unknown :' ))
for i in range(n):
    coordi[i][0] = float(input('Enter x coordinate of station %d :' %(i+1)))
    coordi[i][1] = float(input('Enter y coordinate of station %d :' %(i+1)))
    n_ann_ppt[i] = float(input('Enter normal annual precipitation of %d :' %(i+1)))
    ppt[i] = float(input('Enter precipitation of %d :' %(i+1)))
    r_x1 += ((ppt[i])*n_ann_ppt_u)/((n_ann_ppt[i])*n)
    d[i] = LA.norm(coordi[i] - coordi_u)
    r1 += ppt[i]/(d[i]**2)
    r2 += 1/(d[i]**2)
print('precipitation at unknown point using normal ratio  :' ,r_x1)
print('precipitation at unknown point using inverse ratio method :' ,r1/r2)
    
        