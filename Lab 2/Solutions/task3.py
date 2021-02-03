# Sample solution for lab task 3 (SIFT robustness to changes in rotation)

import cv2
import math
import numpy as np

from compute_sift import SiftDetector

import sys
# Parameters for SIFT initializations such that we find only 25% of keypoints
params = {
    'n_features': 0,    #update this to 100, to view only 100 keypoints
    'n_octave_layers': 3,
    'contrast_threshold': 0.1,  # updated threshold.This value will vary for different images to view 25% of keypoints.
    'edge_threshold': 10,
    'sigma': 1.6
}

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    rot_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
    h, w = image.shape[:2]

    return cv2.warpAffine(image, rot_matrix, (w, h))


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    height, width = image.shape[:2]
    center = height // 2, width // 2

    return center


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    image = cv2.imread("Image1.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = SiftDetector(params=params)

    # Store SIFT keypoints of original image in a Numpy array
    kp1, des1 = sift.detector.detectAndCompute(gray, None)
    OrigKeypoints = cv2.drawKeypoints(gray, kp1, None)
    print("Number of SIFT features: ", len(kp1))
    cv2.imshow("Original keypoints", OrigKeypoints)


    # center of image points. 'img_center' is in (Y, X) order.
    img_center = get_img_center(gray)
    x_coord = img_center[1]
    y_coord = img_center[0]

    # Degrees with which to rotate image
    angle = 90

    # Rotate image
    rotate_gray = rotate(gray, x_coord, y_coord, angle)

    # Compute SIFT features for rotated image
    kp2, des2 = sift.detector.detectAndCompute(rotate_gray, None)
    kp_gray = cv2.drawKeypoints(rotate_gray, kp2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    cv2.imshow("Rotated keypoints", kp_gray)

    # Apply ratio test
    """This test rejects poor matches by computing the ratio between the best and second-best match. 
    If the ratio is below some threshold, the match is discarded as being low-quality."""
    good = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn
    result = cv2.drawMatchesKnn(
        gray, kp1,
        rotate_gray, kp2,
        good, None, flags=2)

    cv2.imshow("Matched points", result)
    cv2.waitKey(0)
