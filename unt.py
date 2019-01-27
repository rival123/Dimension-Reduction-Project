# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:56:35 2018

@author: Rahul
"""

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
data_name = whosmat('IndP_Trn_Tst_Data_10Sample.mat')
loadmat('IndP_Trn_Tst_Data_10Sample.mat')[data_name[3][0]]
tr_data = loadmat('IndP_Trn_Tst_Data_10Sample.mat')[data_name[3][0]]
tr_lab = loadmat('IndP_Trn_Tst_Data_10Sample.mat')[data_name[2][0]]
te_data = loadmat('IndP_Trn_Tst_Data_10Sample.mat')[data_name[1][0]]
te_lab = loadmat('IndP_Trn_Tst_Data_10Sample.mat')[data_name[0][0]]
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
ind1 = []
for i1 in range(tr_lab.shape[0]):
    ind1.append(tr_lab[i1][0]) 
tr_lab1 = np.array(ind1)
ind1 = []
for i1 in range(te_lab.shape[0]):
    ind1.append(te_lab[i1][0]) 
te_lab1 = np.array(ind1)
def ldaf(data,lab,k2):
    mean_vecs = []
    dim = data.shape[1]
    for label in range(1,np.unique(tr_lab).shape[0]):
        mean_vecs.append(np.mean(data[lab == label], axis = 0))
        print('MV %s: %s\n' % (label,mean_vecs[label-1]))
    S_W = np.zeros((dim,dim))
    for label,mv in zip(range(1,np.unique(tr_lab).shape[0]),mean_vecs):
        class_scatter = np.zeros((dim,dim))
        for row in data[lab == label]:
            row,mv = row.reshape(dim,1), mv.reshape(dim,1)
            class_scatter +=(row - mv).dot((row - mv).T)
        S_W += class_scatter
    print('within class matrix: %sx%s' %(S_W.shape[0],S_W.shape[1]))
    #S_W = np.zeros((dim,dim))
    #for label,mv in zip(range(1,np.unique(tr_lab).shape[0]),mean_vecs):
    #    class_scatter = np.cov(data[lab == label].T)
     #   S_W += class_scatter
    #print('scaled within class matrix :' ,S_W.shape)
    mean_overall = np.mean(data,axis = 0)
    S_B = np.zeros((dim,dim))
    for i,mean_vec in enumerate(mean_vecs):
        n = data[lab == i+1,:].shape[0]
        mean_vec = mean_vec.reshape(dim,1)
        mean_overall = mean_overall.reshape(dim,1)
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    print('between scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_vals.shape, eigen_vecs.shape
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eigen_pairs:
        print(i[0])
    W = np.empty(shape = [dim,0])
    for i in range(0,k2):
        W = np.hstack((W, eigen_pairs[i][1].reshape(dim,1)))
    print('Matrix W:\n', W.real)
    f = tr_data.dot(W)
    return f,W
tr_data_lda, red_fac = ldaf(tr_data,tr_lab1,k1)
te_data_lda = te_data.dot(red_fac)
#clf = LDA()
#fit1 = clf.fit(tr_data,tr_lab)
#tr_data_lda1 = fit1.transform(tr_data)
#te_data_lda1 = fit1.transform(te_data)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.00001, C = 1)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_lda,tr_lab1,te_data_lda,te_lab1)
print('accuracy of svm model ud:',acc)
#acc1 = svmclassifier(tr_data_lda1,tr_lab,te_data_lda1,te_lab)
#print('accuracy of svm model pd:',acc1)


        