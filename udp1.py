# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:53:32 2018

@author: ASUS
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.io import loadmat
from scipy.io import whosmat
import numpy as np
import matplotlib as mlt
import pandas as pd
import matplotlib.mlab as mlab
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import linalg as LA
from sklearn.decomposition import PCA
import math
from basicalgos import svmclassifier
from basicalgos import knearestneighbors
from basicalgos import adjacency_laplac
n = input('Enter 1 for Indian pines, 2 for PaviaU, 3 for Botswana, 4 for salinas:')
if n == '1':
    x = 'Indian_pines.mat'
    y = 'Indian_pines_gt.mat'
if n == '2':
    x = 'PaviaU.mat'
    y = 'PaviaU_gt.mat'
if n == '3':
    x = 'Botswana.mat'
    y = 'Botswana_gt.mat'
if n == '4':
    x = 'Salinas.mat'
    y = 'Salinas_gt.mat'
from basicalgos import load_display
img, gt1 = load_display(x,y)
from basicalgos import test_train
typ = input('Enter 0 for number of samples and 1 for percentage of samples for training:')
val = input('Enter the value:')
val = int(val)
k = input('Enter number of dimensions:')
k = int(k)
neigh1 = input('Enter number of neighbors:')
neigh1 = int(neigh1)
tr_data, tr_lab, te_data, te_lab = test_train(img,gt1,typ,val)
def UDP_code(data, gt, k1, k2):
    h_1 = np.zeros((len(data),len(data)))
    h_2 = np.zeros((len(data),len(data)))
    s_list = []
    s1_list = []
    for i in range(len(data)):
        k_near_xi = knearestneighbors(data[i], data, k1)
        s = 0
        s1 =0
        for j in range(len(data)):
            k_near_xj = knearestneighbors(data[j], data, k1)
            if (np.any(k_near_xi == data[j]) & np.any(k_near_xj == data[i])):
                dist = LA.norm(data[i] - data[j])
                h_1[i][j] = math.exp(-(dist**2)/0.5)
            else:
                h_1[i][j] = 0
            h_2[i][j] = 1 - h_1[i][j]
            s += h_1[i][j]
            s1 += h_2[i][j]
        s_list.append(s)
        s1_list.append(s1)
    d_1 = np.diag(s_list)
    l_1 = d_1 - h_1
    s_l = (1/2)*(1/np.size(data))*(1/np.size(data))*(((data.T).dot(l_1)).dot(data))
    print('local scatter matrix:',s_l)
    d_2 = np.diag(s1_list)
    l_2 = d_2 - h_2
    s_n = (1/2)*(1/np.size(data))*(1/np.size(data))*(((data.T).dot(l_2)).dot(data))
    print('non local scatter matrix:',s_n)
    U,s,Vt = LA.svd(LA.inv(s_l).dot(s_n))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    W = np.empty(shape = [data.shape[1],0])
    for i in range(0,k2):
        W = np.hstack((W, eig_pairs[i][1].reshape(data.shape[1],1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_udp, red_fac = UDP_code(tr_data,tr_lab,neigh1,k)
te_data_udp = te_data.dot(red_fac)
print('Accuracy of udp:',svmclassifier(tr_data_udp, tr_lab,te_data_udp,te_lab))
    
    
            
                
    
    