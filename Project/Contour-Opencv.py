# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:39:53 2020

@author: Kovid
"""

from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import matplotlib.pyplot as plt
rng.seed(12345)
def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 1)
    
    
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    k = 0
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        # cv.drawContours(drawing, contours, i, color) #Kovid
        # ellipse
        if c.shape[0] > 20:
            cv.ellipse(drawing, minEllipse[i], color, 1)
            k += 1
        # #rotated rectangle
        # #Kovid
        # box = cv.boxPoints(minRect[i])
        # box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        # cv.drawContours(drawing, [box], 0, color)

    print('Number of contours=',len(contours))
    print('Number of contours printed=',k)
    plt.figure(figsize=(10,10))
    plt.imshow(drawing)
    plt.show()

N = 3 
size = (N, N)
kernel = (3,3)
src = cv.imread('5.tif')
src = cv.imread('89.tif')

src[src > 1] = 255
src[src < 1] = 0

# Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.blur(src_gray, (3,3))


# src = cv.morphologyEx(src, cv.MORPH_OPEN, (30,30))
# src = cv.morphologyEx(src, cv.MORPH_GRADIENT, (5,5))
# src_gray = cv.erode(src_gray,kernel,iterations = 10)
# src_gray = cv.dilate(src_gray,kernel,iterations = 10)
# src_gray = cv.bitwise_not(src_gray)

src = cv.erode(src,kernel,iterations = 10)
src = cv.dilate(src,kernel,iterations = 10)
src_gray = cv.bitwise_not(src)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, size) #cv.MORPH_RECT, cv.MORPH_ELLIPSE, cv.MORPH_CROSS
print('TOPHAT')
tophat = cv.morphologyEx(src, cv.MORPH_TOPHAT, kernel)
plt.imshow(tophat, cmap='gray')
plt.show()

print('TOPHAT - Morph_open')
kernel1 = np.ones((3,3),np.uint8)
O = cv.morphologyEx(tophat, cv.MORPH_OPEN, kernel1)
plt.imshow(O, cmap='gray')
plt.show()    


#Original Image
plt.figure(figsize=(10,10))
plt.imshow(src, cmap='gray')
plt.show()

#Contour Image
plt.figure(figsize=(10,10))
plt.imshow(src_gray, cmap='gray')
plt.show()
max_thresh = 255
thresh = 500 # initial threshold
thresh_callback(thresh)