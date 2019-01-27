# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 21:41:56 2018

@author: Rahul
"""
#%%
from spectral import *
import spectral.io.aviris as aviris
from scipy.io import loadmat
import numpy as np
import matplotlib as mlt
data = loadmat('Indian_pines.mat')['indian_pines_corrected']
view = imshow(data,(29,19,9))
print(view) 
gt = loadmat('Indian_pines_gt.mat')['indian_pines_gt']
view = imshow(classes=gt)
view = imshow(data, (30,20,10), classes=gt)
view.set_display_mode('overlay')
view.class_alpha = 0.5
save_rgb('sample.jpg', data, [1, 1, 140])
save_rgb('gt1.jpg', gt, colors=spy_colors)
##image = data.img()
mlt.pyplot.plot(data[1,1,:])
##data[1,1,:] = aviris.read_aviris_bands('C:\python_3\sample data\92AV3C.spc')
mlt.pyplot.plot(data[1,1,:])
import spectral
spectral.settings.WX_GL_DEPTH_SIZE = 16
pc = principal_components(data)
xdata = pc.transform(data)
w = view_nd(xdata[:,:,:15], classes=gt)
#%%
view_cube(data, bands=[1, 1, 140])
#%%
for i in range(100,125):
    mlt.pyplot.plot(data[i,i+1,:])
#%%    