# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:30:25 2018

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
tr_data, tr_lab, te_data, te_lab = test_train(img,gt1,typ,val)
neigh1 = input('Enter number of neighbors:')
neigh1 = int(neigh1)
neighlab = input('Enter number of neighbors for fuzzy algo:')
neighlab = int(neighlab)
k = input('Enter number of dimensions you want: ')
k = int(k)
def udp_sc_code(data,lab,k1,k2,k3):
    clss = np.unique(lab)
    dim = data.shape[1]
    euc_d = np.zeros((data.shape[0],data.shape[0]))
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
            
    c = np.zeros((dim,dim))
    for cl in clss:
        cls_size = np.size(mlab.find(lab == cl))
        c_p = np.zeros((dim,dim))
        cls_size2 = np.size(mlab.find(lab != cl))
        newcl = list(filter(lambda x : x != cl, clss))
        for xi in data[lab == cl]:
            c_p1 = np.zeros((dim,dim))
            for xj in data[lab != cl]:
                xi1, xj1 = xi.reshape(dim,1), xj.reshape(dim,1)
                print(xi)
                print(xj)
                c_p1 += (h_2[list(np.all(data == xi, axis = 1)).index(True)][list(np.all(data == xj, axis = 1)).index(True)])*(xi1-xj1).dot((xi1-xj1).T)/(cls_size*cls_size2)
            c_p += c_p1
        c += c_p
    print('C :', c)
    a = np.zeros((dim,dim))
    for cl in clss:
        cls_size = np.size(mlab.find(lab == cl))
        a_p = np.zeros((dim,dim))
        for xi in data[lab == cl]:
            a_p1 = np.zeros((dim,dim))
            for xj in data[lab == cl]:
                xi1, xj1 = xi.reshape(dim,1), xj.reshape(dim,1)
                a_p1 += (h_1[list(np.all(data == xi, axis = 1)).index(True)][list(np.all(data == xj, axis = 1)).index(True)])*(xi1-xj1).dot((xi1-xj1).T)/(cls_size*cls_size)
            a_p += a_p1
        a += a_p
    print('A :',a)
    t = c + a
    print('T :', t)
    U,s,Vt = LA.svd(LA.inv(t).dot(c))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    W = np.empty(shape = [dim,0])
    for i in range(0,k3):
        W = np.hstack((W, eig_pairs[i][1].reshape(dim,1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_udp_sc, red_fac = udp_sc_code(tr_data,tr_lab,neigh1,neighlab,k)
te_data_udp_sc = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_udp_sc,tr_lab,te_data_udp_sc,te_lab)
print('accuracy of svm model ud:',acc)