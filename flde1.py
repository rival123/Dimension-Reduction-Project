# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:07:20 2018

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
neighlab = input('Enter number of neighbors for fuzzy algo:')
neighlab = int(neighlab)
tr_data, tr_lab, te_data, te_lab = test_train(img,gt1,typ,val)
def flde_code(data, lab, k1, k2, k3):
    from basicalgos import knearestneighbors
    clss = np.unique(lab)
    dim = data.shape[0]
    euc_d = np.zeros((dim,dim))
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j:
                euc_d[i][j] = math.inf
            else:
                euc_d[i][j] = LA.norm(data[i] - data[j])
    sorted_eud = np.sort(euc_d,axis = 0)
    knear_lab = np.empty(shape = [len(data),k2])
    for j in range(len(data)):
        sort_ind = np.argsort(euc_d[:,j])
        knear_lab[j] = lab[sort_ind[:k2]]
    
    u = np.zeros((len(clss), len(data)))
    for i in range(len(clss)):
        for j in range(len(data)):
            nij = sum(knear_lab[j] == clss[i])
            if lab[j] == clss[i]:
                u[i][j] = 0.51+0.49*(nij/k2)
            else:
                u[i][j] = 0.49*(nij/k2)
    
    H_1 = np.zeros((len(data),len(data)))
    H_2 = np.zeros((len(data),len(data)))
    s_list = []
    s1_list = []
    for i in range(len(data)):
        k_near_xi = knearestneighbors(data[i], data, k1)
        s = 0
        for j in range(len(data)):
            k_near_xj = knearestneighbors(data[j], data, k1)
            if (np.any(k_near_xi == data[j]) & np.any(k_near_xj == data[i]) & lab[i]==lab[j]):
                dist = LA.norm(data[i] - data[j])
                H_1[i][j] = (math.exp(-(dist**2)/0.5))*(u[list(clss).index(lab[i])][j])*(u[list(clss).index(lab[j])][i])
            else:
                H_1[i][j] = 0
            s += H_1[i][j]
        s_list.append(s)
    D_1 = np.diag(s_list)
    L_1 = D_1 - H_1
    S_l = (1/2)*(1/np.size(data))*(1/np.size(data))*(((data.T).dot(L_1)).dot(data))
    print('local scatter matrix:',S_l)
    for i in range(len(data)):
        k_near_xi = knearestneighbors(data[i], data, k1)
        s1 = 0
        for j in range(len(data)):
            k_near_xj = knearestneighbors(data[j], data, k1)
            if(np.any(k_near_xi == data[j]) & np.any(k_near_xj == data[i]) & lab[i] != lab[j]):
                dist = LA.norm(data[i] - data[j])
                H_2[i][j] = (1 - math.exp(-(dist**2)/0.5))*(u[list(clss).index(lab[i])][j])*(u[list(clss).index(lab[j])][i])
            elif(np.any(k_near_xi != data[j]) | np.any(k_near_xj != data[i]) & lab[i] != lab[j]):
                H_2[i][j] = (u[list(clss).index(lab[i])][j])*(u[list(clss).index(lab[j])][i])
            s1 += H_2[i][j]
        s1_list.append(s1)
    D_2 = np.diag(s1_list)
    L_2 = D_2 - H_2
    S_n = (1/2)*(1/np.size(data))*(1/np.size(data))*(((data.T).dot(L_2)).dot(data))
    print('non local scatter matrix:',S_n)
    U,s,Vt = LA.svd(LA.inv(S_l).dot(S_n))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    W = np.empty(shape = [data.shape[1],0])
    for i in range(0,k3):
        W = np.hstack((W, eig_pairs[i][1].reshape(data.shape[1],1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_flde, red_fac = flde_code(tr_data,tr_lab,neigh1,neighlab,k)
te_data_flde = te_data.dot(red_fac)
print('Accuracy of udp:',svmclassifier(tr_data_flde, tr_lab,te_data_flde,te_lab))
        
        
        