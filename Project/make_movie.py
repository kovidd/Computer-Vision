# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:13:10 2020

@author: Kovid
"""

import imutils 
import numpy as np
import cv2
import math
import glob, os
import matplotlib.pyplot as plt

def movie(image_folder):
    name = image_folder.replace('C:\\Users\\Kovid\\Documents\\UNSW\\2020 Term 2\\COMP9517\\Project\\', '')
    name = name.replace('\\', '_')
    name = name.replace(' ', '_')
    path = 'C:/Users/Kovid/Documents/UNSW/2020 Term 2/COMP9517/Project/Videos/'
    num_files = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])   
    video_name = f'{name} {num_files+1}.avi'
    print('video_name=',video_name)
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    framespsec = 5
    os.chdir('C:/Users/Kovid/Documents/UNSW/2020 Term 2/COMP9517/Project/Videos/')
    video = cv2.VideoWriter(video_name, 0, framespsec, (width,height)) #(video_name, 0, fps, (width,height))
    
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    global cwd
    os.chdir(cwd)

sequence = ['Output']
sequence = ['Output/01']

cwd = os.getcwd()
# for folder in folders:
for seq in sequence:
    os.chdir('C:/Users/Kovid/Documents/UNSW/2020 Term 2/COMP9517/Project/'+seq+'/')
    print(os.getcwd())
    movie(os.getcwd())