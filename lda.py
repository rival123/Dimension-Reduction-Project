# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 19:41:01 2018

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
#var_req = input('enter minimum % variance required after feature reduction: ')
#var_req = float(var_req)
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
##new_tr_data,evec = pcaf(tr_data,var_req)
##new_te_data = np.matmul(te_data, evec)
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
def ldaf(tr_data,tr_lab,k2):
    mean_vectors = []
    clss = np.unique(tr_lab)
    dim = tr_data.shape[1]
    for cl in clss :
        mean_vectors.append(np.mean(tr_data[tr_lab==cl], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
     ###  scipy.linalg.solve python
    S_W = np.zeros((dim,dim))
    for cl,mv in zip(clss, mean_vectors):
        class_sc_mat = np.zeros((dim,dim))                   
        for row in tr_data[tr_lab == cl]:
            row, mv = row.reshape(dim,1), mv.reshape(dim,1) 
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat
    overall_mean = np.mean(tr_data, axis=0)
    S_B = np.zeros((dim,dim))
    for i,mean_vec in enumerate(mean_vectors):
        n = tr_data[tr_lab==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(dim,1) 
        overall_mean = overall_mean.reshape(dim,1) 
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    print('between-class Scatter Matrix:\n', S_B)
    U,s,Vt = LA.svd(LA.inv(S_W).dot(S_B))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    for i in range(len(eig_vals)):
        eigvec_sc = eig_vecs[:,i].reshape(dim,1)   
        #print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
        #print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    for i,j in enumerate(eig_pairs):
        print('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
    W = np.empty(shape = [dim,0])
    for i in range(0,k2):
        W = np.hstack((W, eig_pairs[i][1].reshape(dim,1)))
    print('Matrix W:\n', W.real)
    f = tr_data.dot(W)
    return f,W
tr_data_lda, red_fac = ldaf(tr_data,tr_lab,k1)
te_data_lda = te_data.dot(red_fac)
#clf = LDA()
#fit1 = clf.fit(tr_data,tr_lab)
#tr_data_lda1 = fit1.transform(tr_data)
#te_data_lda1 = fit1.transform(te_data)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_lda,tr_lab,te_data_lda,te_lab)
print('accuracy of svm model ud:',acc)
#acc1 = svmclassifier(tr_data_lda1,tr_lab,te_data_lda1,te_lab)
#print('accuracy of svm model pd:',acc1)

