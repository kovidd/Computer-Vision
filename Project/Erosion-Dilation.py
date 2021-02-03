# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:29:50 2020

@author: Kovid
"""

from __future__ import print_function
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
erosion_size = 0
max_elem = 2
max_kernel_size = 21

def erosion(src, s, t):
    erosion_size = s
    val_type = t
    if val_type == 0:
        erosion_type = cv.MORPH_RECT
    elif val_type == 1:
        erosion_type = cv.MORPH_CROSS
    elif val_type == 2:
        erosion_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    erosion_dst = cv.erode(src, element)
    # global src
    # src = erosion_dst
    plt.imshow(erosion_dst)
    plt.show()
    return erosion_dst
    
def dilatation(src, s, t):
    dilatation_size = s
    val_type = t
    if val_type == 0:
        dilatation_type = cv.MORPH_RECT
    elif val_type == 1:
        dilatation_type = cv.MORPH_CROSS
    elif val_type == 2:
        dilatation_type = cv.MORPH_ELLIPSE
    element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dilatation_dst = cv.dilate(src, element)
    plt.imshow(dilatation_dst)
    plt.show()
    return dilatation_dst

src = cv.imread('5.tif')
size = 20
dt = 2
src = erosion(src, 10, dt)
src = dilatation(src, size, dt)