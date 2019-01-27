# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 16:27:35 2018

@author: Rahul
"""

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
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import linalg as LA
from sklearn.decomposition import PCA
import math
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
def knearestneighbors(point,datasets,k):
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
n_unsuper = input('number of unsupervised data points :')
n_unsuper = int(n_unsuper)
tr_datau = te_data[np.random.choice(te_data.shape[0],n_unsuper,replace=False), :]
tr_data_new = np.vstack((tr_data,tr_datau))
neighlab = input('number of nearest neighbors for fuzzy algo: ')
neighlab= int(neighlab)
neighdt = input('number of nearest data points :')
neighdt = int(neighdt)
neigh2 = input('Number of nearest neighbors in laplacian matrix: ')
neigh2 = int(neigh2)
k1 = input('Enter number of dimensions you want: ')
k1 = int(k1)
def ssflsc_code(data, data_new, lab, k2, k3, k4, neigh2):
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
    c_td = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(clss)):
        c_tdi1 = np.zeros((data.shape[1],data.shape[1]))
        cls_size = np.size(data[lab == clss[i]])
        for xi in data[lab == clss[i]]:
            c_tdi = np.zeros((data.shape[1],data.shape[1]))
            k_nearest = knearestneighbors(xi,data[lab != clss[i]],k3)
            for xj in k_nearest:
                xi , xj1 = xi.reshape(data.shape[1],1), xj.reshape(data.shape[1],1)
                c_tdi += u[i][np.where(np.all(data == xj, axis = 1))]*((xi - xj1).dot((xi - xj1).T))/(cls_size*k3)
            c_tdi1 += c_tdi
        c_td += c_tdi1
    print('C tilda :', c_td)
    a_td = np.zeros((data.shape[1],data.shape[1]))
    for i in range(len(clss)):
        a_tdi1 = np.zeros((data.shape[1],data.shape[1]))
        cls_size = np.size(data[lab == clss[i]])
        for xi in data[lab == clss[i]]:
            a_tdi = np.zeros((data.shape[1],data.shape[1]))
            k_nearest = knearestneighbors(xi,data[lab == clss[i]],k3)
            for xj in k_nearest:
                xi , xj1 = xi.reshape(data.shape[1],1), xj.reshape(data.shape[1],1)
                a_tdi += u[i][np.where(np.all(data == xj, axis = 1))]*((xi - xj1).dot((xi - xj1).T))/(cls_size*k3)
            a_tdi1 += a_tdi
        a_td += a_tdi1     
    print('A tilda :', a_td)
    t_td = c_td + a_td
    print('T_tilda :', t_td)
    matrices = adjacency_laplac(data_new, neigh2)
    print('adjacency matrix :',matrices[0])
    print('degree matrix :',matrices[1])
    print("laplacian's matrix :",matrices[2])
    denom = t_td + 0.5*(((data_new.T).dot(matrices[2])).dot(data_new))
    U,s,Vt = LA.svd(LA.inv(denom).dot(c_td))
    V = Vt.T
    eig_vals = s**2
    eig_vecs = V
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    print('Eigenvalues in decreasing order:\n')
    for i in eig_pairs:
        print(i[0])
    W = np.empty(shape = [data.shape[1],0])
    for i in range(0,k4):
        W = np.hstack((W, eig_pairs[i][1].reshape(data.shape[1],1)))
    print('Matrix W:\n', W.real)
    f = data.dot(W)
    return f,W
tr_data_ssflsc, red_fac = ssflsc_code(tr_data,tr_data_new,tr_lab,neighlab,neighdt,k1,neigh2)
te_data_ssflsc = te_data.dot(red_fac)
def svmclassifier(traindata, trainlabel,testdata,testlabel):
    clf = svm.SVC(kernel = "linear",gamma=0.1, C = 10000000)
    clf.fit(traindata, trainlabel)
    prdict = clf.predict(testdata)
    return accuracy_score(testlabel,prdict)
acc = svmclassifier(tr_data_ssflsc,tr_lab,te_data_ssflsc,te_lab)
print('accuracy of svm model ud:',acc)
        
            
    