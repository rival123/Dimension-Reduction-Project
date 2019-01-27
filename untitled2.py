# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 14:28:42 2018

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
n = input('Enter 1 for Indian pines and 2 for PaviaU:')
if n == '1':
    x = 'Indian_pines.mat'
    y = 'Indian_pines_gt.mat'
if n == '2':
    x = 'PaviaU.mat'
    y = 'PaviaU_gt.mat'
def load_display(filename, groundtruth):
    data_name = whosmat(filename)
    d = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', d)
    gt_name = whosmat(groundtruth)
    gt1 = loadmat(groundtruth)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt1)
    d1 = d.reshape((d.shape[0]*d.shape[1],d.shape[2]))
    return d1 , gt1
data1 = load_display(x,y)
img = data1[0]
gt1 = data1[1]
typ = input('Enter 0 for number of samples and 1 for percentage of samples for training:')
val = input('Enter the value:')
val = int(val)
neigh1 = input('number of nearest neighbors: ')
neigh1= int(neigh1)
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
def test_train(data, gt, typ1, val1):
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
        cls_size = np.size(mlab.find(gtarray[0,:] == i))
        print('size of class',i, cls_size)
        if cls_size <= 100:
            gtarray = np.delete(gtarray,mlab.find(gtarray[0,:] == i),axis = 1)
            gtloc = np.setdiff1d(gtloc,mlab.find(gt_cor == i))
            continue
        if typ1 == '0':
            val2 = val1
        if typ1 == '1':
            val2 = val1*cls_size/100 
        val2 = round(val2)
        rtemp = np.random.choice(gtarray[1,mlab.find(gtarray[0,:] == i)], val2, replace=False)
        rtempl = rtemp.tolist()
        rsamp = rsamp + rtempl
        print(len(rsamp))
    train_index = np.array(rsamp)
    print('train index:',train_index)
    train_data = data[train_index]
    print('train data:' ,train_data)
    train_data= train_data.astype(float)
    #min_max = MinMaxScaler()
    #scaled_data = min_max.fit_transform(train_data)
    scaler = preprocessing.StandardScaler().fit(train_data)
    scaled_data = scaler.transform(train_data)
    test_index = np.setdiff1d(gtloc,train_index)
    print('test index:',test_index)
    test_data = data[test_index]
    print('test data:',test_data)
    test_data= test_data.astype(float)
    #scaled_test = min_max.transform(test_data)
    scaled_test = scaler.transform(test_data)
    trlab = gtcor[train_index]
    print('train label:', trlab)
    telab = gtcor[test_index] 
    print('test label', telab)
    return scaled_data, trlab, scaled_test, telab
k = test_train(img,gt1,typ,val)
tr_data = k[0]
tr_lab = k[1]
te_data = k[2]
te_lab = k[3]
def FLDA_code(data,lab,k,k2):
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
    knear_lab = np.empty(shape = [len(data),k])
    for j in range(len(data)):
        sort_ind = np.argsort(euc_d[:,j])
        knear_lab[j] = lab[sort_ind[:k]]
    
    u = np.zeros((len(clss), len(data)))
    m = np.zeros((len(clss), data.shape[1]))
    for i in range(len(clss)):
        sum_m = np.empty(shape = [data.shape[1]])
        sum_u = 0
        for j in range(len(data)):
            nij = sum(knear_lab[j] == clss[i])
            if lab[j] == clss[i]:
                u[i][j] = 0.51+0.49*(nij/k)
            else:
                u[i][j] = 0.49*(nij/k)
            sum_m += u[i][j]*data[j]
            sum_u += u[i][j]
        m[i] = sum_m/sum_u
    
    fsw = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(clss)):
        fsw_i = np.zeros((data.shape[1],data.shape[1]))        
        for xj in data[lab == clss[i]]:
            xj1, m1 = xj.reshape(data.shape[1],1), m[i].reshape(data.shape[1],1) 
            fsw_i += u[i][np.where(np.all(data == xj, axis = 1))]*((xj1 - m1).dot((xj1 - m1).T))
        fsw += fsw_i 
    overall_mean = np.mean(data, axis=0)
    fsb = np.zeros((data.shape[1],data.shape[1]))     
    for i in range(len(clss)):
        fsb_i = np.zeros((data.shape[1],data.shape[1])) 
        for j in range(len(data)):
            overall_mean, m2 = overall_mean.reshape(data.shape[1],1), m[i].reshape(data.shape[1],1)  
            fsb_i += u[i][j]*((m2 - overall_mean).dot((m2 - overall_mean).T))
        fsb += fsb_i
        
    U,s,Vt = LA.svd(LA.inv(fsw).dot(fsb))
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
    f = tr_data.dot(W)
    return f,W
tr_data_flda, red_fac = FLDA_code(tr_data,tr_lab,neigh1,k1)
te_data_flda = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "linear",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_flda,tr_lab,te_data_flda,te_lab)
print('accuracy of svm model ud:',acc)


        
            