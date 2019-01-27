# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:57:41 2018

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
if n == '2':
    x = 'PaviaU.mat'
def load_display(filename):
    data_name = whosmat(filename)
    d = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', d)
    slic = filename[:-4] + '_gt' + filename[-4:]
    gt_name = whosmat(slic)
    gt1 = loadmat(slic)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt1)
    d1 = d.reshape((d.shape[0]*d.shape[1],d.shape[2]))
    return d1 , gt1
data1 = load_display(x)
img = data1[0]
gt1 = data1[1]
typ = input('Enter 0 for number of samples and 1 for percentage of samples for training:')
val = input('Enter the value:')
val = int(val)
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
k1 = input('enter number of dimensions you want :')
k1 = int(k1)
def dlpp_code(data,lab,k2):
    clss = np.unique(lab)
    dim = data.shape[0]
    w = np.zeros((dim,dim))
    d = np.zeros((dim,dim))
    edge_coordinates = [0,0]
    for cl in clss:
        dim1 = data[lab == cl].shape[0]
        w_i = np.zeros((dim1, dim1))
        s_list = []
        for j in range(len(data[lab == cl])):
            s = 0
            for k in range(len(data[lab == cl])):
                dist = LA.norm(data[lab == cl][j] - data[lab == cl][k])
                w_i[j][k] = math.exp(-(dist**2)/0.5)
                s += w_i[j][k]
            s_list.append(s)
        d_i = np.diag(s_list)
        slicer = tuple(slice(edge, edge+i) for edge, i in zip(edge_coordinates, w_i.shape))
        w[slicer] = w_i
        d[slicer] = d_i
        edge_coordinates = [x+w_i.shape[0] for x in edge_coordinates]
    print('within class weight matrix :',w)
    print('D :',d)
    l = d-w
    print('laplacian matrix L :',l)
    f = np.empty(shape =[0,data.shape[1]])
    for cl in clss:
        fi = np.zeros((1,data.shape[1]))
        dim2 = data[lab == cl].shape[0]
        for vecs in data[lab == cl]:
            fi += vecs/dim2
        f = np.vstack((f,fi))
    s1_list = []
    b = np.zeros((len(f),len(f)))
    for i in range(len(f)):
        s1 = 0
        for j in range(len(f)):
            dist1 = LA.norm(f[i] - f[j])
            b[i][j] = math.exp(-(dist1**2)/0.5)
            s1 += b[i][j]
        s1_list.append(s1)
    e = np.diag(s1_list)
    print('between class weight matrix:',b)
    print('E :',e)
    h = e-b
    print('laplacian matrix H:' ,h)
    s_w_l = ((data.T).dot(l)).dot(data)
    s_b_l = ((f.T).dot(h)).dot(f)
    U,s,Vt = LA.svd(LA.inv(s_w_l).dot(s_b_l))
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
    new_d = tr_data.dot(W)
    return new_d,W
tr_data_dlpp, red_fac = dlpp_code(tr_data,tr_lab,k1)
te_data_dlpp = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 100000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_dlpp,tr_lab,te_data_dlpp,te_lab)
print('accuracy of svm model ud:',acc)    