# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 18:31:43 2018

@author: Rahul
"""

import spectral
from spectral import *
import spectral.io.aviris as aviris
from scipy.io import loadmat
from scipy.io import whosmat
import numpy as np
import matplotlib as mlt
def display_data(filename):
    data_name = whosmat(filename)
    data = loadmat(filename)[data_name[0][0]]
    print ('dictionary of pixels of data:', data)
    view = imshow(data,(55,100,75))
    print('image by three rows:',view) 
    slic = filename[:-4] + '_gt' + filename[-4:]
    gt_name = whosmat(slic)
    gt = loadmat(slic)[gt_name[0][0]]
    print ('dictionary of ground truth data:',gt)
    view = imshow(classes=gt)
    print ('image of ground truth data:',view)
    view = imshow(data, (55,100,75), classes=gt)
    print ('merging original with the ground truth image:',view)
    view.set_display_mode('overlay')
    view.class_alpha = 0.5
    save_rgb('sample.jpg', data, [55,100,75])
    print ('data image saved as sample')
    save_rgb('gt1.jpg', gt, colors=spy_colors)
    print ('ground truth data image saved as gt1')
    print ('plot of pixels vs band of a point:')
    mlt.pyplot.plot(data[110,123,:])
            ##spectral.settings.WX_GL_DEPTH_SIZE = 16
            ##pc = principal_components(data)
            ##xdata = pc.transform(data)
            ##w = view_nd(xdata[:,:,:15], classes=gt)
            ##view_cube(data, bands=[1, 1, 140])
    print ('plot of pixels vs band for different points')
    for i in range(100,125):
        mlt.pyplot.plot(data[i,i+1,:])
        #%%
display_data('PaviaU.mat')
     