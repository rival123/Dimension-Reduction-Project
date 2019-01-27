# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:13:53 2018

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
        if typ1 == '0':
            val2 = val1
        if typ1 == '1':
            val2 = val1*cls_size/100 
        val2 = round(val2)
        rtemp = np.random.choice(gtarray[1,mlab.find(gtarray[0,:] == i)], val2, replace=False)
        rtempl = rtemp.tolist()
        rsamp = rsamp + rtempl
        print(len(rsamp))
    print(rsamp)
    train_index = np.array(rsamp)
    print('train index:',train_index)
    train_data = data[train_index]
    print('train data:' ,train_data)
    train_data= train_data.astype(float)
    min_max = MinMaxScaler()
    scaled_data = min_max.fit_transform(train_data)
    test_index = np.setdiff1d(gtloc,train_index)
    print('test index:',test_index)
    test_data = data[test_index]
    print('test data:',test_data)
    test_data= test_data.astype(float)
    scaled_test = min_max.transform(test_data)
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
var_req = input('enter minimum % variance required after feature reduction: ')
var_req = float(var_req)
def pcaf(data,p):
    U,s,Vt = LA.svd(tr_data)
    w = s**2
    e_count = s.shape[0]
    var =0
    k = 0
    for i in range(0,e_count-1):
        var = var + w[i]
        k = k+1
        if var/np.sum(w) >= p/100:
            break
    V = Vt.T
    V1 = V[:,0:k]
    trdata = np.matmul(data, V1)
    return trdata, V1
new_tr_data,evec = pcaf(tr_data,var_req)
new_te_data = np.matmul(te_data, evec)
pca1 = PCA(n_components = 19)
new_tr_data = pca1.fit_transform(tr_data)
new_te_data = pca1.transform(te_data)

def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    print('accuracy of svm model:',accuracy_score(testlabel,prdict))
svmclassifier(new_tr_data,tr_lab, new_te_data, te_lab)
def knnclassifier(traindata, trainlabel,testdata,testlabel):
    #neigh = KNeighborsClassifier(n_neighbors=3)
    neigh = neighbors.KNeighborsClassifier()
    neigh.fit(traindata , trainlabel)
    prdict = neigh.predict(testdata)
   # print(metrics.classification_report(testlabel, prdict))
    print('accuracy of knn model:',accuracy_score(testlabel,prdict))
knnclassifier(new_tr_data,tr_lab, new_te_data, te_lab)
def ldaclassifier (traindata, trainlabel,testdata,testlabel):
    clf = LDA()
    clf.fit(traindata , trainlabel)
    prdict = clf.predict(testdata)
    #print('accuracy of lda model:',accuracy_score(testlabel,prdict))
    print('accuracy of lda model:',clf.score(testdata,testlabel))
ldaclassifier(new_tr_data, tr_lab, new_te_data, te_lab)
    
    