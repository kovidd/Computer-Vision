from scipy.spatial import distance as dist
from collections import OrderedDict
from collections import deque
import cv2
#import imutils
import time
import os
import glob
import sys
from PIL import Image, ImageOps
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage.io
import copy
import math

class Tracker():
    '''
    #Simon: set maxDisappeared=0 means the object will be deregistered immediately once not detected.
    It is safe to ignore all code related to self.disappeared
    '''
    def __init__(self, algo, maxDisappeared=0):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.centroid_paths = OrderedDict() #Simon: added to trace trajectory of each object
        self.objects_contours = OrderedDict() #Simon: keep a record or the contour of each object
        self.objects_under_mitosis = OrderedDict() #Simon: recording if the cell is under mitosis
        self.objects_area_when_mitosis = OrderedDict() #Simon: record a list of contour area for cells undergoes mitosis
		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        
        #choose algos depends on input
        if algo == 'Centroid':
            self.update = self.centroid_update
        else:
            print('algo undefined')
        
    def register(self, centroid, contour):
		# when registering an object we use the next available object
		# ID to store the centroid
        #Simon: initialize centroid_path for that object when registering
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.centroid_paths[self.nextObjectID] = [centroid]
        self.objects_contours[self.nextObjectID] = contour
        self.objects_under_mitosis[self.nextObjectID] = False #cell is assumed not to be under mitosis once first detected
        self.objects_area_when_mitosis[self.nextObjectID] = 0
        self.nextObjectID += 1
        
    def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.centroid_paths[objectID]
        del self.objects_contours[objectID]
        del self.objects_under_mitosis[objectID]
        del self.objects_area_when_mitosis[objectID]
        
    def centroid_update(self, rects, contours):
        '''
        Simon: probably modify the code below for using another tracker than centroid tracking
        Most lines here are just comments
        '''
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
				# missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
            return self.objects
		# initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
		# centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], contours[i])
		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
        else:
			# grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value is at the *front* of the index
			# list
            rows = D.min(axis=1).argsort()
            #print(rows)
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            #print(cols)
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
				# column value before, ignore it
				# val
                if row in usedRows or col in usedCols:
                    continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
                objectID = objectIDs[row]
                new_centroid = inputCentroids[col]
                new_contour = contours[col]
                self.objects[objectID] = new_centroid
                self.objects_contours[objectID] = new_contour
                self.disappeared[objectID] = 0
                self.centroid_paths[objectID].append(new_centroid)
                if self.objects_under_mitosis[objectID] == False:
                    self.objects_under_mitosis[objectID] = mitosis_detection(objectID, new_contour)
                    if self.objects_under_mitosis[objectID] == True:
                        self.objects_area_when_mitosis[objectID] = cv2.contourArea(new_contour)
                else:
                    area_when_mitosis = self.objects_area_when_mitosis[objectID]
                    result = mitosis_finished(objectID, new_contour, area_when_mitosis)
                    #if mitosis finished, assign the cell cell as new cell, rather than retaining the same ID as parent
                    if result == True:
                        self.register(new_centroid, new_contour)
                        self.deregister(objectID)
				# indicate that we have examined each of the row and
				# column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
            else:
                pass
            for col in unusedCols:
                self.register(inputCentroids[col], contours[col])
		# return the set of trackable objects
        return self.objects, self.centroid_paths, self.objects_contours, self.objects_under_mitosis
'''
define a function for easier image display
'''
def image_display(message, input_img):
    print(message)
    cv2.imshow(message, input_img)
    k = cv2.waitKey(0)
    return
def diplay_frame_with_wait(frame, waiting):
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(waiting) & 0xFF
    return key


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
function for reteiving contours based on a given segmented image and input
'''
def get_contour_using_mask(image,seg):
    seg = copy.deepcopy(seg)
    seg = seg.astype(np.uint8)
    seg2 = copy.deepcopy(seg)
    #num of segmented cells = highest pixel value
    num_obj = seg.max()
    
    #loop through all segmented cells, add to list of contours
    all_contours = []
    for m in range(1,num_obj+1):
        seg = copy.deepcopy(seg2)
        #mask out all other pixels except for the current cell object
        seg[np.where(seg != m)] = 0
        # Find external contours, each loop there should only be one contour added, 
        _,contour,_ = cv2.findContours(seg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # Iterate over each found contour to add to list
        #simon: contour should only be one element each, but sometimes it returns 2 elements, still debugging
        for contour_element in contour:
            all_contours.append(contour_element)
    return all_contours

'''
function for filtering out useless contours
'''
def contour_processing(all_contours):
    out_contours = []
    min_area_requirement = 500 #hard-coded number for now, feel free to change
    #filter out contours that are too small
    for contour in all_contours:
        area = cv2.contourArea(contour)
        if area < min_area_requirement:
            continue
        out_contours.append(contour)
    return out_contours
'''
function for obtaining all bounding boxes from the contours
'''
def get_rect_from_contour(all_contours):
    out_rects = []
    for contour in all_contours:
        rect = cv2.minAreaRect(contour)
        out_rects.append(rect)
    return out_rects
'''
function for obtain bbox from rectangles
'''
def get_bbox_from_rectangle(all_rects):
    out_bboxes = []
    for rect in all_rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        out_bboxes.append(box)
    return out_bboxes
'''
function for formatting bboxes
'''
def formatting_bbox(all_bbox):
    out_rects = []
    for bbox in all_bbox:
        start_x = min(i[0] for i in bbox)
        end_x = max(i[0] for i in bbox)
        start_y = min(i[1] for i in bbox)
        end_y = max(i[1] for i in bbox)
        bbox_tuple = (start_x, start_y, end_x, end_y)
        out_rects.append(bbox_tuple)
    return out_rects
'''
function for drawing the contour on input image
'''
def draw_contour_on_input(image, all_contours):
    image = copy.deepcopy(image)
    #define 2 colors, only the first color is used for now
    RGBforLabel = { 1:(0,0,255), 2:(0,255,255) }
    for i,c in enumerate(all_contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg.shape, np.uint8)
        cv2.drawContours(mask,[c],-1,255, -1)
    
        # Get appropriate colour for this label, not used for now
        #mean,_,_,_ = cv2.mean(seg, mask=mask)
        #label = 2 if mean > 1.0 else 1
        
        colour = RGBforLabel.get(1)
    
        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(image,[c],-1,colour,1)
    return image

'''
function for drawing all bbox on image
'''
def draw_bbox_on_input(image, all_bbox, status):
    image = copy.deepcopy(image)
    #define 2 colors, only the first color is used for now
    RGBforLabel = { False:(0,255,255), True:(50,100,200) }
    
    #loop through each rectangles
    for i,box in enumerate(all_bbox):
        # get the actual 4 corners of bbox from the recrangle
        
        # Get appropriate colour for this label, not used for now
        #mean,_,_,_ = cv2.mean(seg, mask=mask)
        #label = 2 if mean > 1.0 else 1
        colour = RGBforLabel.get(status[i])
        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(image,[box],0,colour,1)
    return image

'''
function for processing user input
'''
def passing_input(input_str):
    if input_str.isdigit():
        return int(input_str)
    elif input_str =='quit':
        return input_str
    else:
        return 'error'

'''
function for showing task3 informations
'''
def show_speed_and_distance(out_image, obj_id, all_paths):
    path = trajectorys[input_id]
    #not abele to show speed if there is only one element
    if len(path) == 1:
        speed = 'No speed'
        total_dist = 0
        net_dist = 0
        confinement_ratio = 'N/A'
    else:
        #otherwise calculate speed using last 2 centroids
        curr_dist = cv2.norm(path[-1] - path[-2], cv2.NORM_L2)
        curr_dist = round(curr_dist,2)
        speed = str(curr_dist) + " pixels/sec"
        net_dist = cv2.norm(path[0] - path[-1], cv2.NORM_L2)
        total_dist = 0
        #loop through path to calculate total dist
        for centre_id in range(len(path)-1):
            start_point = path[centre_id]
            end_point = path[centre_id+1]
            dist = cv2.norm(start_point - end_point, cv2.NORM_L2)
            total_dist += dist
        confinement_ratio = total_dist/net_dist
        confinement_ratio = str(round(confinement_ratio,2))
    
    #print info to console
    print("^^^^^^Cell-info^^^^^^^^")
    print('Cell id: %d' % obj_id)
    print('Speed: %s' % speed)
    print('Total_dist: %d' % total_dist)
    print('Net_dist: %d' % net_dist)
    print('Confinement: %s' % confinement_ratio)
    print("^^^^^^^^^^^^^^^^^^^^^^^")
    'Finish code here if need to display the info on the image, rahter than printing on console'        
    #cv2.putText(out_img, speed, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    #image_display('Cell_info', out_img)
    
    return
    
'''
Simon: Parameter/logic needs to be changed for each dataset
function to determine a object detected has mitosis just begins
Condition need to satisfy: 
    -Circular shaped
    -higher peak intensity
    -area > min-required area
'''
def mitosis_detection(obj_id, contour):
    max_allowed_compactness = 1.45
    #min_intensity_requirement = 121
    min_intensity_requirement = frame_mean_intensity+2
    min_area = 7500
    under_mitosis = False
    contour_area = cv2.contourArea(contour)
    contour_perimeter = cv2.arcLength(contour,True)
    #calculate compactness to determine circularity (i.e. if the contour is circle shaped)
    compactness = contour_perimeter**2 / (4*math.pi * contour_area)
    #find the average intensity of the cell, using the original input file.
    mask = np.zeros(main.shape,np.uint8)
    cv2.drawContours(mask,[contour],0,255,-1)
    pixelpoints = np.transpose(np.nonzero(mask))
    mean_val = cv2.mean(main,mask = mask)[0]
    if compactness < max_allowed_compactness and mean_val > min_intensity_requirement and contour_area > min_area:
        under_mitosis = True
    
    return under_mitosis

def mitosis_finished(obj_id, contour, old_area):
    #mitosis is finished if the contour area dropped under certain value
    area_ratio_threshold = 0.8
    contour_area = cv2.contourArea(contour)
    if (contour_area/old_area)<area_ratio_threshold:
        return True
    else:
        return False

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
    
    img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
    
    return img_in

'''
contour detection for 3 channel image.
Parameter needs to be tuned
'''
def find_size(img_in):
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    gray[gray != 0] = 255
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    thresh_3d = np.repeat(thresh[:, :, np.newaxis], 3, axis=2)

    lower = np.array([5, 5, 5])
    upper = np.array([255, 255, 255])
    shapeMask = cv2.inRange(thresh_3d, lower, upper)

    # find the contours in the mask
    _, cnts, _ = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnts = imutils.grab_contours(cnts)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img_in, cnts, -1, (0, 255, 0), 1)
    return img_in

def do_watershed(img_in):
    img_in = Image.open(f1)
    img_array = np.array(ImageOps.grayscale(img_in))
    distance = ndi.distance_transform_edt(img_array)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=img_array)
    
    return labels
def do_canny_edge(img_in):
    edges = cv2.Canny(img_in,80,150)
    plt.subplot(121),plt.imshow(img_in,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    return edges

"========================Main=================================================="
# construct the argument parse and parse the arguments
DIC_folder = '/home/simon/COMP9517/project/data2/DIC-C2DH-HeLa/01/*.tif'
DIC_mask_folder = '/home/simon/COMP9517/project/data2/DIC-C2DH-HeLa/01_RES/*.tif'
fluo_folder = "/home/simon/COMP9517/project/Fluo-N2DL-HeLa/Sequence 3/*.tif"
Phc_folder = "/home/simon/COMP9517/project/PhC-C2DL-PSC/Sequence 2/*.tif"
#os.chdir("Fluo-N2DL-HeLa")
video_path = 'Videos/PhC-C2DL-PSC_Sequence_1.avi'
video_path = '/home/simon/COMP9517/project/Videos/PhC-C2DL-PSC_Sequence_1.avi'
# initialize a dictionary that maps strings to their corresponding
"===========Unsed Code========================="
# # OpenCV object tracker implementations
# OPENCV_OBJECT_TRACKERS = {
# 	"csrt": cv2.TrackerCSRT_create,
# 	"kcf": cv2.TrackerKCF_create,
# 	"boosting": cv2.TrackerBoosting_create,
# 	"mil": cv2.TrackerMIL_create,
# 	"tld": cv2.TrackerTLD_create,
# 	"medianflow": cv2.TrackerMedianFlow_create,
# 	"mosse": cv2.TrackerMOSSE_create
# }
# # initialize OpenCV's special multi-object tracker
# tracker_type = 'medianflow'
# trackers = cv2.MultiTracker_create()
"======================================"


img_files = glob.glob(DIC_folder)
img_files.sort()
mask_files = glob.glob(DIC_mask_folder)
mask_files.sort()

#wait time decides gap-time between each frame. =0 means manual key-press required
wait_time = 0
centroid_tracker = Tracker('Centroid')
#play all frames in the folder
for k in range(0,len(img_files)):
    f1 = img_files[k]
    mask1 = mask_files[k]
    print(f1)
    #open input and segmented image
    main = cv2.imread(f1,cv2.IMREAD_GRAYSCALE)
    out_img = copy.deepcopy(main)
    seg = cv2.imread(mask1, -1)
    out_img = cv2.cvtColor(main,cv2.COLOR_GRAY2BGR)
    frame_mean_intensity = cv2.mean(main)[0]
    #get contours from segmented image
    contours = get_contour_using_mask(main,seg)
    #post-processing contour to remove unneeded ones (i.e. too small)
    contours = contour_processing(contours)
    #get rectangle and bbox from using contour information
    rects = get_rect_from_contour(contours)
    bboxes= get_bbox_from_rectangle(rects)
    #formatting the bboxes so it can be used for tracker update. See the /above code for detail of update
    formatted_rects = formatting_bbox(bboxes)
    
    objects, trajectorys, obj_contours, mitosis_status = centroid_tracker.update(formatted_rects,contours)
    
    '''
    Simon: The following plotting code will work as long as 2 dictionaries of centroid and paths are returned
    '''
    # loop over the tracked objects to draw centroid and IDs
    for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(out_img, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(out_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    for (objectID, paths) in trajectorys.items():
        #print((objectID, paths))
        for i in range(1, len(paths)):
            if paths[i - 1] is None or paths[i] is None:
                continue
            cv2.line(out_img, tuple(paths[i - 1]), tuple(paths[i]), (255, 100, 255), 1)    
    
    #return a image with all the bbox shown
    
    #re-compute all the bbox and decide color using updated info
    status = list(mitosis_status.values())
    ordered_contours = list(obj_contours.values())
    ordered_rects = get_rect_from_contour(ordered_contours)
    ordered_bboxes= get_bbox_from_rectangle(ordered_rects)
    
    bbox_image = draw_bbox_on_input(out_img, ordered_bboxes, status)
    #return a image with all the contours shown
    contour_image = draw_contour_on_input(out_img, contours)
    
    #print the total number of cell and mitosis cell in console:
    print('>>>>>>>>>>>>>>>>>>')
    print('Number of cells detected: %d' % len(objects))
    print('Number of mitosis cell: %d' % sum(mitosis_status.values()))
    print('>>>>>>>>>>>>>>>>>>')
    '''
    uncomment the code below for plotting the original segmented image
    '''
    #unsplit_mask = skimage.io.imread(mask1)
    #plt.imshow(unsplit_mask)
    #plt.show()
    
    #do action depend on the value of key pressed
    return_key = diplay_frame_with_wait(bbox_image, wait_time)
    if return_key == 0xFF:
        pass
    elif return_key == ord("q"):
        print('quitting')
        break
    elif return_key == ord("c"):
        while True:
            #the following code handles task3
            user_input = input('Type the cell-id to show its info: ')
            #convert 
            input_id = passing_input(user_input)
            if(input_id == 'error'):
                print('invalid input, try again')
                continue
            elif(input_id == 'quit'):
                print('quitting cell selection')
                break
            #check if the input integer is valid
            if input_id in objects.keys():
                #print out the task3 information on the image
                show_speed_and_distance(bbox_image, input_id, trajectorys)
                pass
                continue
            else:
                print('invalid object id, try again')
                continue               
            break
    else:
        pass
        #print('lalala')
        
cv2.destroyAllWindows()