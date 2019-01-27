# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 09:19:48 2018

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
def load_display(filename, groundtruth):
    from scipy.io import loadmat
    from scipy.io import whosmat
    data_name = whosmat(filename)
    d = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', d)
    gt_name = whosmat(groundtruth)
    gt1 = loadmat(groundtruth)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt1)
    d1 = d.reshape((d.shape[0]*d.shape[1],d.shape[2]))
    return d1 , gt1
def knearestneighbors(point,datasets,k):
    from numpy import linalg as LA
    import pandas as pd
    distss = []
    for xk in datasets:
        dist = LA.norm(point - xk)
        distss.append(dist)
    distance_frame = pd.DataFrame(data={"distance": distss, "idx": distss.index})
    distance_frame.sort_values('distance',inplace = True)
    if distance_frame.iloc[0]['distance'] == 0:
        k_nearest_index = distance_frame.iloc[1:k+1]['idx']
    else:
        k_nearest_index = distance_frame.iloc[0:k]['idx']
    k_nearest = datasets[k_nearest_index.index]
    return(k_nearest)
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
def adjacency_laplac(datasets, nei):
    m = np.zeros((len(datasets),len(datasets)))
    s_list = []
    for j in range(len(datasets)):
        k_nearest = knearestneighbors(datasets[j], datasets, nei)
        s=0
        for k in range(len(datasets)):
            if np.any(k_nearest == datasets[k]):
                dist = LA.norm(datasets[j] - datasets[k])
                m[j][k] = math.exp(-(dist**2)/0.5)
            else:
                m[j][k] = 0
            s += m[j][k]
        s_list.append(s)
    d = np.diag(s_list)
    l = d-m
    return m, d, l
def distancess(x):
    from numpy import linalg as LA
    return LA.norm(x)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "rbf",gamma=0.1, C = 100000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
