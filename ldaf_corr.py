# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 19:14:35 2018

@author: Rahul
"""
from sklearn import preprocessing
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
from scipy import linalg as lg
from sklearn.decomposition import PCA
from scipy import stats
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
    a = np.mean(train_data, axis = 0)
    b = np.std(train_data, axis = 0)
    #scaler = preprocessing.StandardScaler().fit(train_data)
    #scaled_data = scaler.transform(train_data)
    #min_max = MinMaxScaler()
    #scaled_data = min_max.fit_transform(train_data)
    sc = np.empty(shape = [train_data.shape[0],train_data.shape[1]])
    for j in range(train_data.shape[0]):
        sc[j] = (train_data[j] - a)/b
    scaled_data = sc
    test_index = np.setdiff1d(gtloc,train_index)
    print('test index:',test_index)
    test_data = data[test_index]
    print('test data:',test_data)
    test_data= test_data.astype(float)
    #scaled_test = min_max.transform(test_data)
    #scaled_test = scaler.transform(test_data)
    sc1 = np.empty(shape = [test_data.shape[0],test_data.shape[1]])
    for j in range(test_data.shape[0]):
        sc1[j] = (test_data[j] - a)/b
    scaled_test = sc1    
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
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
def ldaf(data,lab,k2):
    mean_vectors = []
    clss = np.unique(tr_lab)
    for cl in clss :
        mean_vectors.append(np.mean(data[lab==cl], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    S_W = np.zeros((220,220))
    for cl,mv in zip(clss, mean_vectors):
        class_sc_mat = np.zeros((220,220))                   
        for row in data[lab == cl]:
            row, mv = row.reshape(220,1), mv.reshape(220,1) 
            class_sc_mat += np.dot((row-mv), (row-mv).T)    
        S_W += class_sc_mat
    print(S_W)
    overall_mean = np.mean(data, axis=0)
    S_B = np.zeros((220,220))
    for i,mean_vec in enumerate(mean_vectors):
        n = data[lab==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(220,1) 
        overall_mean = overall_mean.reshape(220,1) 
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    print('between-class Scatter Matrix:\n', S_B)
    w, vl, vr = lg.eig(S_B, S_W, left = True, right = True)
    w1 = w.real
    eig_pairs = [(np.abs(w1[i]), vr[:,i]) for i in range(len(w1))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    print('Variance explained:\n')
    eigv_sum = sum(w1)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    W = np.empty(shape = [220,0])
    for i in range(0,k2):
        W = np.hstack((W, eig_pairs[i][1].reshape(220,1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_lda, red_fac = ldaf(tr_data,tr_lab,k1)
te_data_lda = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_lda,tr_lab,te_data_lda,te_lab)
print('accuracy of svm model ud:',acc)