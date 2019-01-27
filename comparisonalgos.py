# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:15:45 2018

@author: Rahul
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
import time
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
typ = input('Enter 0 for number of samples and 1 for percentage of samples for training:')
val = input('Enter the value:')
val = int(val)
k = input('Number of dimensions:')
k = int(k)
neigh1 = input('number of nearest neighbors: ')
neigh1= int(neigh1)
n_unsuper = input('number of unsupervised data points :')
n_unsuper = int(n_unsuper)
neigh2 = input('Number of nearest neighbors in laplacian matrix: ')
neigh2 = int(neigh2)
neighlab = input('number of nearest neighbors for fuzzy algo: ')
neighlab= int(neighlab)
pca_acc = np.zeros((5))
lda_acc = np.zeros((5))
sc_acc = np.zeros((5))
lsc_acc = np.zeros((5))
dlpp_acc = np.zeros((5))
sslsc_acc = np.zeros((5))
flda_acc = np.zeros((5))
flsc_acc = np.zeros((5))
ssflsc_acc = np.zeros((5))
udp_acc = np.zeros((5))
flde_acc = np.zeros((5))
udp_sc_acc = np.zeros((5))
udp_flsc_acc = np.zeros((5))
ttest = np.zeros((5))
for i in range(5):
    start_time = time.clock()
    from basicalgos import test_train
    tr_data, tr_lab, te_data, te_lab = test_train(img,gt1,typ,val)
    tr_datau = te_data[np.random.choice(te_data.shape[0],n_unsuper,replace=False), :]
    tr_data_new = np.vstack((tr_data,tr_datau))
    ttest[i] = time.clock() - start_time    

    from dimredtech import ssflsc_code
    tr_data_ssflsc, red_fac = ssflsc_code(tr_data,tr_data_new,tr_lab,neighlab,neigh1,k,neigh2)
    te_data_ssflsc = te_data.dot(red_fac)
    ssflsc_acc[i] = svmclassifier(tr_data_ssflsc,tr_lab,te_data_ssflsc,te_lab)
    
    from dimredtech import sslsc_code
    tr_data_sslsc, red_fac = sslsc_code(tr_data,tr_data_new,tr_lab,neigh1,neigh2,k)
    te_data_sslsc = te_data.dot(red_fac)
    sslsc_acc[i] = svmclassifier(tr_data_sslsc,tr_lab,te_data_sslsc,te_lab)
'''    
    from dimredtech import flsc_code
    tr_data_flsc, red_fac = flsc_code(tr_data,tr_lab,neighlab,neigh1,k)
    te_data_flsc = te_data.dot(red_fac)
    flsc_acc[i] = svmclassifier(tr_data_flsc,tr_lab,te_data_flsc,te_lab)
    
    start_time = time.clock()
    from dimredtech import UDP_flsc_code
    tr_data_udp_flsc, red_fac = UDP_flsc_code(tr_data,tr_lab,neigh2,neighlab,neigh1,k)
    te_data_udp_flsc = te_data.dot(red_fac)
    udp_flsc_acc[i] = svmclassifier(tr_data_udp_flsc,tr_lab,te_data_udp_flsc,te_lab)
    tuflsc[i] = time.clock() - start_time
    start_time = time.clock()

    from dimredtech import pcafnumber
    new_tr_data,evec = pcafnumber(tr_data,k)
    new_te_data = np.matmul(te_data, evec)
    pca_acc[i] =  svmclassifier(new_tr_data,tr_lab,new_te_data, te_lab)
    
    from dimredtech import ldaf
    tr_data_lda,red_fac = ldaf(tr_data,tr_lab,k)
    te_data_lda = te_data.dot(red_fac)
    lda_acc[i] = svmclassifier(tr_data_lda,tr_lab,te_data_lda, te_lab)
    
    from dimredtech import sc_code
    new_tr_data, red_fac = sc_code(tr_data,tr_lab,k)
    new_te_data = te_data.dot(red_fac)
    sc_acc[i] =  svmclassifier(new_tr_data,tr_lab,new_te_data, te_lab)
    
    from dimredtech import lsc_code
    tr_data_lsc, red_fac = lsc_code(tr_data,tr_lab,neigh1,k)
    te_data_lsc = te_data.dot(red_fac)
    lsc_acc[i] = svmclassifier(tr_data_lsc,tr_lab,te_data_lsc,te_lab)
    
    from dimredtech import udp_sc_code
    tr_data_udp_sc, red_fac = udp_sc_code(tr_data,tr_lab,neigh2,neighlab,k)
    te_data_udp_sc = te_data.dot(red_fac)
    udp_sc_acc[i] = svmclassifier(tr_data_udp_sc,tr_lab,te_data_udp_sc,te_lab)
    
    from dimredtech import flde_code
    tr_data_flde, red_fac = flde_code(tr_data,tr_lab,neigh2,neighlab,k)
    te_data_flde = te_data.dot(red_fac)
    flde_acc[i] = svmclassifier(tr_data_flde, tr_lab,te_data_flde,te_lab)
    
    from dimredtech import UDP_code
    tr_data_udp, red_fac = UDP_code(tr_data,tr_lab,neigh2,k)
    te_data_udp = te_data.dot(red_fac)
    udp_acc[i] = svmclassifier(tr_data_udp, tr_lab,te_data_udp,te_lab)
    
    from dimredtech import dlpp_code
    tr_data_dlpp, red_fac = dlpp_code(tr_data,tr_lab,neigh2,k)
    te_data_dlpp = te_data.dot(red_fac)
    dlpp_acc[i] = svmclassifier(tr_data_dlpp,tr_lab,te_data_dlpp,te_lab)
        
    from dimredtech import FLDA_code
    tr_data_flda, red_fac = FLDA_code(tr_data,tr_lab,neighlab,k)
    te_data_flda = te_data.dot(red_fac)
    flda_acc[i] = svmclassifier(tr_data_flda,tr_lab,te_data_flda,te_lab)
    

    

'''
    

'''    
print('avg accuracy of pca :',np.mean(pca_acc))
print('avg accuracy of lda :',np.mean(lda_acc))
print('avg accuracy of sc :',np.mean(sc_acc))
print('avg accuracy of lsc :',np.mean(lsc_acc))
print('avg accuracy of dlpp :',np.mean(dlpp_acc))       
print('avg accuracy of flda :',np.mean(flda_acc))   
''' 

print('avg accuracy of udp_flsc :',np.mean(udp_flsc_acc))
print('avg accuracy of flsc :',np.mean(flsc_acc)) 
print('avg accuracy of ssflsc :',np.mean(ssflsc_acc))
print('avg accuracy of sslsc :',np.mean(sslsc_acc))  

