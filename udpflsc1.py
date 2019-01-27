# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:06:40 2018

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
neigh3 = input('Enter number of neighbors for lsc:')
neigh3 = int(neigh3)
neighlab = input('Enter number of neighbors for fuzzy algo:')
neighlab = int(neighlab)
tr_data, tr_lab, te_data, te_lab = test_train(img,gt1,typ,val)
def UDP_flsc_code(data, lab, k1, k2,k3,k4):
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
                
    h_1 = np.zeros((len(data),len(data)))
    h_2 = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        k_near_xi = knearestneighbors(data[i], data, k1)
        for j in range(len(data)):
            k_near_xj = knearestneighbors(data[j], data, k1)
            if (np.any(k_near_xi == data[j]) & np.any(k_near_xj == data[i])):
                dist = LA.norm(data[i] - data[j])
                h_1[i][j] = math.exp(-(dist**2)/0.5)*(u[list(clss).index(lab[i])][j])
            else:
                h_1[i][j] = 0
            h_2[i][j] = (u[list(clss).index(lab[i])][j]) - h_1[i][j]
            
    c_td = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(clss)):
        c_tdi1 = np.zeros((data.shape[1],data.shape[1]))
        cls_size = np.size(data[lab == clss[i]])
        for xi in data[lab == clss[i]]:
            c_tdi = np.zeros((data.shape[1],data.shape[1]))
            k_nearest = knearestneighbors(xi,data[lab != clss[i]],k3)
            for xj in k_nearest:
                xi1 , xj1 = xi.reshape(data.shape[1],1), xj.reshape(data.shape[1],1)
                c_tdi += (h_2[list(np.all(data == xi, axis = 1)).index(True)][list(np.all(data == xj, axis = 1)).index(True)])*((xi1 - xj1).dot((xi1 - xj1).T))/(cls_size*k3)
            c_tdi1 += c_tdi
        c_td += c_tdi1
    print('C tilda :', c_td)
    a_td = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(clss)):
        a_tdi1 = np.zeros((data.shape[1],data.shape[1]))
        cls_size = np.size(data[lab == clss[i]])
        for xi in data[lab == clss[i]]:
            a_tdi = np.zeros((data.shape[1],data.shape[1]))
            k_nearest = knearestneighbors(xi,data[lab == clss[i]],k3)
            for xj in k_nearest:
                xi1 , xj1 = xi.reshape(data.shape[1],1), xj.reshape(data.shape[1],1)
                a_tdi += (h_1[list(np.all(data == xi, axis = 1)).index(True)][list(np.all(data == xj, axis = 1)).index(True)])*((xi1 - xj1).dot((xi1 - xj1).T))/(cls_size*k3)
            a_tdi1 += a_tdi
        a_td += a_tdi1     
    print('A tilda :', a_td)
    t_td = c_td + a_td
    print('T_tilda :', t_td)
            
    U,s,Vt = LA.svd(LA.inv(t_td).dot(c_td))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    W = np.empty(shape = [data.shape[1],0])
    for i in range(0,k4):
        W = np.hstack((W, eig_pairs[i][1].reshape(data.shape[1],1)))
    print('Matrix W:\n', W.real)
    f = tr_data.dot(W)
    return f,W
tr_data_udp_flsc, red_fac = UDP_flsc_code(tr_data,tr_lab,neigh1,neighlab,neigh3,k)
te_data_udp_flsc = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "linear",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_udp_flsc,tr_lab,te_data_udp_flsc,te_lab)
print('accuracy of svm model ud:',acc)
        