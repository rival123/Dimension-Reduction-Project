# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:00:49 2018

@author: Rahul
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import spectral
from spectral import *
import spectral.io.aviris as aviris
from scipy.io import loadmat
from scipy.io import whosmat
import numpy as np
import matplotlib as mlt
import pandas as pd
import matplotlib.mlab as mlab
def preprocessdata(filename):
    data_name = whosmat(filename)
    data = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', data)
    slic = filename[:-4] + '_gt' + filename[-4:]
    gt_name = whosmat(slic)
    gt = loadmat(slic)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt)
    d = data.reshape((data.shape[0]*data.shape[1],200))
    d = d.astype(float)
    min_max = MinMaxScaler()
    scaled = min_max.fit_transform(d)
    print('data normalized:',scaled)
    gt_cor = gt.reshape((gt.shape[0]*gt.shape[1],))
    print(gt_cor)
    gtloc = mlab.find(gt_cor > 0)
    gtcl = gt_cor[gtloc]
    gtarray = np.vstack((gtcl, gtloc))
    gtr = gtarray[:, gtarray[0,:].argsort()]
    clss = np.unique(gtarray[0,:])
    rsamp = []
    gtcor = np.array(gt_cor)
    for i in clss:
        print('size of class',i,np.size(mlab.find(gtarray[0,:] == i)))
        rtemp = np.random.choice(gtarray[1,mlab.find(gtarray[0,:] == i)], 10, replace=False)
        rtempl = rtemp.tolist()
        rsamp = rsamp + rtempl
        print(len(rsamp))
    print(rsamp)
    train_index = np.array(rsamp)
    print('train index:',train_index)
    train_data = d[train_index]
    print('train data:' ,train_data)
    test_index = np.delete(gtloc,train_index)
    print('test index:',test_index)
    test_data = d[test_index]
    print('test data:',test_data)
    #trlab= []
    #for j in train_index:
        #trlab = trlab + gtarray[0,mlab.find(gtarray[1,:] == j)].tolist()
    #print(trlab)
    trlab = gtcor[train_index]
    print('train label:', trlab)
    telab = gtcor[test_index] 
    print('test label', telab)     
    #train_label = gtarray[1,mlab.find(gtarray[0,:]
    #test_label = gtarray[0,test_index] 
        
    
preprocessdata('Indian_pines.mat')
            
        