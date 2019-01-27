# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:42:14 2018

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
    mean_vectors = []
    dim = data.shape[1]
    clss = np.unique(tr_lab)
    for cl in clss :
        ind1 = []
        for i1 in range(lab.shape[0]):
            if lab[i1][0] == cl:
                ind1.append(i1)
        mean_vectors.append(np.mean(data[ind1], axis=0))
        print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
    S_W = np.zeros((dim,dim))
    for cl,mv in zip(clss, mean_vectors):
        class_sc_mat = np.zeros((dim,dim)) 
        ind1 = []
        for i1 in range(lab.shape[0]):
            if lab[i1][0] == cl:
                ind1.append(i1)                  
        for row in data[ind1]:
            row, mv = row.reshape(dim,1), mv.reshape(dim,1) 
            class_sc_mat += np.dot((row-mv), (row-mv).T)    
        S_W += class_sc_mat
        print(S_W)
    print(S_W)
    overall_mean = np.mean(data, axis=0)
    S_B = np.zeros((dim,dim))
    for i,mean_vec in enumerate(mean_vectors):
        ind1 = []
        for i1 in range(lab.shape[0]):
            if lab[i1][0] == i+1:
                ind1.append(i1)        
        n = data[ind1,:].shape[0]
        mean_vec = mean_vec.reshape(dim,1) 
        overall_mean = overall_mean.reshape(dim,1) 
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
    W = np.empty(shape = [dim,0])
    for i in range(0,k2):
        W = np.hstack((W, eig_pairs[i][1].reshape(dim,1)))
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
acc = svmclassifier(tr_data_lda,tr_lab1,te_data_lda,te_lab1)
print('accuracy of svm model ud:',acc)