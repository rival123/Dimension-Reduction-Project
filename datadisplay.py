# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 22:19:18 2018

@author: Rahul
"""
import spectral
from spectral import *
import spectral.io.aviris as aviris
from scipy.io import loadmat
import numpy as np
import matplotlib as mlt
def display_data(x):
        s1 = "Indian pines"
        s2 = "paviaU"
        if x.lower() == s1.lower():
            data = loadmat('Indian_pines.mat')['indian_pines']
            gt = loadmat('Indian_pines_gt.mat')['indian_pines_gt']
        elif x.lower() == s2.lower():
            data = loadmat('PaviaU.mat')['paviaU']
            gt = loadmat('PaviaU_gt.mat')['paviaU_gt']
        else:
            print ("your input does not match the desired input")
            quit()
        print ('dictionary of pixels of data:', data)
        view = imshow(data,(55,100,75))
        print('image by three rows:',view) 
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
display_data('paviau')