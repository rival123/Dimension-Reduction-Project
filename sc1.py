# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:48:51 2018

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
from sklearn.metrics import confusion_matrix
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
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
def sc_code(data,lab,k2):
    clss = np.unique(lab)
    dim = data.shape[1]
    c = np.zeros((dim,dim))
    for cl in clss:
        cls_size = np.size(mlab.find(lab == cl))
        c_p = np.zeros((dim,dim))
        cls_size2 = np.size(mlab.find(lab != cl))
        newcl = list(filter(lambda x : x != cl, clss))
        for xi in data[lab == cl]:
            c_p1 = np.zeros((dim,dim))
            for xj in data[lab != cl]:
                xi, xj = xi.reshape(dim,1), xj.reshape(dim,1)
                c_p1 += (xi-xj).dot((xi-xj).T)/(cls_size*cls_size2)
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
                xi, xj = xi.reshape(dim,1), xj.reshape(dim,1)
                a_p1 += (xi-xj).dot((xi-xj).T)/(cls_size*cls_size)
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
    for i in range(0,k2):
        W = np.hstack((W, eig_pairs[i][1].reshape(dim,1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_sc, red_fac = sc_code(tr_data,tr_lab,k1)
te_data_sc = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "linear",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    accu = accuracy_score(testlabel,prdict)
    return accu, prdict
acc, pred_lab = svmclassifier(tr_data_sc,tr_lab,te_data_sc,te_lab)
conf_mat = confusion_matrix(te_lab, pred_lab)
conf_mat = np.array(conf_mat)
TP = np.diag(conf_mat)
FP = np.sum(conf_mat, axis=0) - TP
FN = np.sum(conf_mat, axis=1) - TP
num_classes = np.size(np.unique(tr_lab))
TN = []
for i in range(num_classes):
    temp = np.delete(conf_mat, i, 0)    # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TN.append(sum(sum(temp)))
prec_class= TP/(TP+FP)
rec_class = TP/(TP+FN)
if type == 1:
    avg_prec = sum(TP)/(sum(TP)+sum(FP)) 
    avg_rec = sum(TP)/(sum(TP)+sum(FN))
else:
    avg_prec = np.mean(prec_class)
    avg_rec = np.mean(rec_class)
specificity = sum(TN)/(sum(TN)+sum(FP))
beta = 1
f1_score = (1+beta**2)*(avg_prec*avg_rec/((beta**2)*avg_prec + avg_rec))
g_measure = math.sqrt(avg_prec*avg_rec)
obs_acc = sum(np.diag(conf_mat)/conf_mat.sum())
expec_acc = 0
for v1 in range(conf_mat.shape[1]):
    expec_acc = expec_acc + (sum(conf_mat[v1])*(sum(conf_mat[:,v1])))/conf_mat.sum()
expec_acc = expec_acc/conf_mat.sum()
k = (obs_acc - expec_acc)/(1-expec_acc)
clss = np.unique(te_lab)
nclss = len(clss)
nrpixelsperclass = np.zeros((nclss,1))
errormatrix = np.zeros((nclss,nclss))
for i in range(nclss):
    indi = mlab.find(te_lab == clss[i])
    nrpixelsperclass[i] = len(indi)
    for j in range(nclss):
        indj = mlab.find(pred_lab == clss[j])
        errormatrix[i,j] = len(np.intersect1d(indi, indj))
diagvector = np.diag(errormatrix)
CA = diagvector/(nrpixelsperclass + np.spacing(1))
AA = np.mean(CA)
OA = sum(pred_lab == te_lab)/len(te_lab)
kA = (errormatrix.sum()*sum(np.diag(errormatrix)) - errormatrix.sum(axis=0).dot((errormatrix.sum(axis =1)).T))/((errormatrix.sum())**2 - errormatrix.sum(axis=0).dot((errormatrix.sum(axis =1)).T))
print('accuracy of svm model ud:',acc)
print('Predicted labels :',prdict)
print('precision :',avg_prec)
print('Recall :',avg_rec)
print('Specificity :',specificity)
print('F1 score :', f1_score)
print('g score :', g_measure)
print('CA :',CA)
print('AA :',AA)
print('OA :',OA)
print('kA :',kA)