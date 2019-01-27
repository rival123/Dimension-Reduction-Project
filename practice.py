# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 12:24:06 2018

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
import pandas
def preprocessdata(filename):
    data_name = whosmat(filename)
    data = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', data)
    slic = filename[:-4] + '_gt' + filename[-4:]
    gt_name = whosmat(slic)
    gt = loadmat(slic)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt)
    data = data.astype(float)
    #gt_scaled = preprocessing.scale(gt)
    #print('scaled ground truth values',gt_scaled) 
    #print('Mean',gt_scaled.mean(axis=0))
    #print('standard deviation',gt_scaled.std(axis=0))                                        
    #gt_normalized = preprocessing.normalize(gt, norm='l1')
    #print(gt_normalized) 
    min_max = MinMaxScaler()
    scaled = min_max.fit_transform()
    print(scaled)
    data = data.astype(float)
    #for j in range(0,data.shape[2]-1):
     #   min_max1 = MinMaxScaler()
      #  scaled_data[:,:,j] = min_max1.fit_transform(data[:,:,j])
    #print (scaled_data)
    visualisation(scaled, data,(55,100,75),data[110,123,:] )
def visualisation(sc_gt, sc_data, bands, point):
    view = imshow(sc_data,bands)
    view = imshow(classes=sc_gt)
    print ('image of ground truth data:',view)
    mlt.pyplot.plot(point)
preprocessdata('Indian_pines.mat')                                      


    