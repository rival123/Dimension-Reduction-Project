# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:49:53 2018

@author: Rahul
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import spectral
from spectral import *
import spectral.io.aviris as aviris
from scipy.io import loadmat
from scipy.io import whosmat
import numpy as np
import matplotlib as mlt
import pandas
def train_and_test(filename):
    data_name = whosmat(filename)
    data = loadmat(filename)[data_name[0][0]]
    slic = filename[:-4] + '_gt' + filename[-4:]
    gt_name = whosmat(slic)
    gt = loadmat(slic)[gt_name[0][0]]
    v = imshow(classes=gt)
    classes = create_training_classes(data, gt)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(data)
    v = imshow(classes=clmap)
    gtresults = clmap * (gt != 0)
    v = imshow(classes=gtresults)
    gterrors = gtresults * (gtresults != gt)
    v = imshow(classes=gterrors)
    #%%
    #dimensionality reduction
    pc = principal_components(data)
    v = imshow(pc.cov)
    pc_0999 = pc.reduce(fraction=0.999)
    len(pc_0999.eigenvalues)
    data_pc = pc_0999.transform(data)
    v = imshow(data_pc[:,:,:3], stretch_all=True)
    classes = create_training_classes(data_pc, gt)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(data_pc)
    clmap_training = clmap * (gt != 0)
    v = imshow(classes=clmap_training)
    training_errors = clmap_training * (clmap_training != gt)
    v = imshow(classes=training_errors)
#%%
    #fischer linear discriminant
    classes = create_training_classes(data, gt)
    fld = linear_discriminant(classes)
    len(fld.eigenvectors)
    data_fld = fld.transform(data)
    v = imshow(data_fld[:, :, :3])
    classes.transform(fld.transform)
    gmlc = GaussianClassifier(classes)
    clmap = gmlc.classify_image(data_fld)
    clmap_training = clmap * (gt != 0)
    v = imshow(classes=clmap_training)
    fld_errors = clmap_training * (clmap_training != gt)
    v = imshow(classes=fld_errors)
#%%
train_and_test('Indian_pines.mat')
    