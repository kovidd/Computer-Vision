# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 18:14:00 2020

@author: Kovid
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def side_by_side(img1, img2):
    print('\nStacking Images')
    # equ = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    stacked = np.hstack((img1,img2)) #stacking images side-by-side
    # Draw line between them
    width, height = stacked.shape[1], stacked.shape[0]
    x1, y1 = int(width/2), 0
    x2, y2 = int(width/2), height
    line_thickness = 2
    stacked_with_line = cv2.line(stacked, (x1, y1), (x2, y2), (255, 255, 255), thickness=line_thickness)
    # plt.figure(figsize=(10,10))
    # plt.imshow(stacked_with_line, cmap = 'gray')
    # plt.show()
    return stacked_with_line

def thresholding(img):
    ##################### Thresholding ####################
    # img = cv2.imread('Fluo-N2DL-HeLa/Sequence 1 ST/man_seg091.tif')
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    # print('\nOtsu\'s thresholding')
    retval2, img1 = cv2.threshold(grayscaled,1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.figure(figsize=(10,10))
    # plt.imshow(img1, cmap = 'gray')
    # plt.show()
    # Otsu's thresholding after Gaussian filtering
    # print('\nOtsu\'s thresholding after Gaussian filtering')
    blur = cv2.GaussianBlur(grayscaled,(3,3),0)
    ret3,img2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    plt.figure(figsize=(10,10))
    plt.imshow(img2, cmap = 'gray')
    plt.show()
    # Adaptive thresholding
    # print('\nAdaptive thresholding')
    img3 = cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,255,0)
    # plt.figure(figsize=(10,10))
    # plt.imshow(img3, cmap = 'gray')
    # plt.show()
    return img1, img2, img3
    ##################### Thresholding ####################    
    
# folder_1 = "Fluo-N2DL-HeLa-Kovid/Bounding Boxes - Input/*.tif"
# folder_2 = "Fluo-N2DL-HeLa-Kovid/Bounding Boxes - ST/*.tif"


# folder_1 = "Output/Track - Contours of trial 10/*.tif"
# folder_2 = "Output/Track - Adaptive 01/*.tif"

folder_1 = "Fluo-N2DL-HeLa-Kovid/Sequence 1/*.tif"
folder_2 = "Fluo-N2DL-HeLa-Kovid/GT/TRA/Sequence 1/*.tif"

img1_files = glob.glob(folder_1)
img1_files.sort()
img2_files = glob.glob(folder_2)
img2_files.sort()

for i in range(len(img1_files)):
    img1 = cv2.imread(img1_files[i])
    img2 = cv2.imread(img2_files[i])
    # otsu1, otsublur1, adathr1 = thresholding(img1)
    # otsu2, otsublur2, adathr2 = thresholding(img2)
    
    result = side_by_side(img1, img2)
    _,name = img1_files[i].split('\\')
    print(name)
    path = 'C:/Users/Kovid/Documents/UNSW/2020 Term 2/COMP9517/Project/Fluo-N2DL-HeLa-Kovid/Compare Segment with GT/'
    # print(os.path.join(path, name))
    cv2.imwrite(os.path.join(path, name), result)

# img1 = cv2.imread('Fluo-N2DL-HeLa-Kovid/Sequence 1/t091.tif')
# img2 = cv2.imread('Fluo-N2DL-HeLa-Kovid/Bounding Boxes - Input/t091.tif')

# img1 = cv2.imread('Fluo-N2DL-HeLa/Sequence 1/t091.tif')
# otsu1, otsublur1, adathr1 = thresholding(img1)
# otsu2, otsublur2, adathr2 = thresholding(img2)

# result = side_by_side(otsu1, otsu2)
# plt.figure(figsize=(10,5))
# plt.imshow(result, cmap = 'gray')
# plt.show()

# result = side_by_side(otsublur1, otsublur2)
# plt.figure(figsize=(10,5))
# plt.imshow(result, cmap = 'gray')
# plt.show()

# result = side_by_side(adathr1, adathr2)
# plt.figure(figsize=(10,5))
# plt.imshow(result, cmap = 'gray')
# plt.show()