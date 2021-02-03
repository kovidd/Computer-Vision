#https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/


from collections import deque
import numpy as np
import argparse
import cv2
#import imutils
import time
import os
import glob
import sys
import glob
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

'''
define a function for easier image display
'''
def image_display(message, input_img):
    print(message)
    cv2.imshow(message, input_img)
    k = cv2.waitKey(0)
    return k

'''
apply to original image from imread()
'''
def normalize(image):
    '''
    This function is to normalize the input grayscale image by
    substracting globle mean and dividing standard diviation for
    visualization. 
    Input:  a grayscale image
    Output: normolized grascale image
    '''
    img = image.copy().astype(np.float32)
    img -= np.mean(img)
    img /= np.linalg.norm(img)
    # img = (img - img.min() )
    img = np.clip(img, 0, 255)
    img *= (1./float(img.max()))
    return (img*255).astype(np.uint8)
'''
min max filtering for normalized image
'''
def min_max_filter(img_in):
    '''
    min & max filtering taken from Ass1
    '''
    neighbourhood_radius = 5
    size = (neighbourhood_radius, neighbourhood_radius)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    #max filter image and display
    img_A= cv2.erode(img_in, kernel)
    image_display('min', img_A)
    
    img_B = cv2.dilate(img_A,kernel)
    image_display('max', img_B)
    
    return img_B
'''
blob detection for 3 channel image. 
Parameter needs to be tuned
'''
def blob_detection(img_in):
    '''
    blob detection
    '''
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByArea = False
    params.filterByConvexity = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_in)
    
    img_in = cv2.drawKeypoints(img_in, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ##Kovid
    minRect = [None]*len(contours)
    box = cv.boxPoints(minRect[i])
    box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv.drawContours(drawing, [box], 0, color)
    
    
    img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
    
    return img_in

'''
contour detection for 3 channel image.
Parameter needs to be tuned
'''
import random as rng
import matplotlib.pyplot as plt
rng.seed(12345)
json_dict = {}
k = 0
def find_size(img_in):
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    gray[gray != 0] = 255
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    thresh_3d = np.repeat(thresh[:, :, np.newaxis], 3, axis=2)

    lower = np.array([5, 5, 5])
    upper = np.array([255, 255, 255])
    shapeMask = cv2.inRange(thresh_3d, lower, upper)

    # find the contours in the mask
    cnts, _  = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_in, cnts, -1, (0, 255, 0), 1)

    #Find bboxes
    threshold = 500    
    src_gray = img_in
    canny_output = cv2.Canny(src_gray, threshold, threshold * 1)
    contours, _  = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
    minRect = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
    # The output of cv2.minAreaRect() is ((x, y), (w, h), angle
    detections_x1y1x2y2 = {}
    box_list = []
    for e in minRect:
        # print('minRect = ', e[0], e[1])    
        box_list.append([e[0][0], e[0][1], e[1][0], e[1][1]])
    
    detections_x1y1x2y2 = box_list
    # print('detections_x1y1x2y2',detections_x1y1x2y2)
    
    global k
    key_name = f'{k}'
    json_dict[key_name]=detections_x1y1x2y2
    k += 1
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)    
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        box = cv2.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(drawing, [box], 0, color)
        # print('box=', box)
    # plt.figure(figsize=(10,10))
    # plt.imshow(drawing)
    # plt.show()
    # return img_in
    return drawing


# construct the argument parse and parse the arguments
# fluo_folder = "Fluo-N2DL-HeLa/Sequence 1/*.tif"
# fluo_folder = "Fluo-N2DL-HeLa/Sequence 1 Test/*.tif"
fluo_folder = "Fluo-N2DL-HeLa/Sequence 1 ST/*.tif"

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
tracker_type = 'medianflow'
trackers = cv2.MultiTracker_create()
img_files = glob.glob(fluo_folder)
img_files.sort()

for f1 in img_files:
    frame = cv2.imread(f1)
    frame = normalize(frame)
    frame = find_size(frame)
    '''
    Standard manual tracking code
    '''
    # check to see if we have reached the end of the stream
    if frame is None:
        print('3')
        break
    # resize the frame (so we can process it faster)
    #frame = imutils.resize(frame, width=1000)
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
    	(x, y, w, h) = [int(v) for v in box]
    	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Finally, show the output frame
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(0) & 0xFF
    # write images to make video
    _,name = f1.split('\\')
    print(name)
    path = 'C:/Users/Kovid/Documents/UNSW/2020 Term 2/COMP9517/Project/Fluo-N2DL-HeLa-Kovid/Bounding Boxes - ST/'
    # print(os.path.join(path, name))
    cv2.imwrite(os.path.join(path, name), frame)

# print('\nJSON:\n',repr(json_dict))
f = open('json - ST.txt', 'w')
f.write(repr(json_dict))
f.close()
# for kk,vv in json_dict.items():
#     print()
#     print(kk, vv)
    
print('Json Length:',len(json_dict))
    