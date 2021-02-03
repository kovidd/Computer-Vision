import cv2

from compute_sift import SiftDetector


# Parameters for SIFT initializations such that we find only 25% of keypoints
params = {
    'n_features': 0,    #update this to 100, to view only 100 keypoints
    'n_octave_layers': 3,
    'contrast_threshold': 0.08,  # updated threshold.This value will vary for different images to view 25% of keypoints.
    'edge_threshold': 10,
    'sigma': 1.6
}


if __name__ == '__main__':
    # 1. Read image
    image = cv2.imread("image2.jpg")

    # 2. Convert image to grayscale
    grayImageOrig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = SiftDetector(params=params)

    # Store SIFT keypoints of original image in a Numpy array
    kp,des1 = sift.detector.detectAndCompute(grayImageOrig, None)

    kp_gray = cv2.drawKeypoints(image, kp, grayImageOrig)
    cv2.imshow("Original keypoints", kp_gray)
    print("Number of SIFT features: ", len(kp))
    cv2.waitKey(100)

    #Upscale the image
    scale_percent = 110  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resizedImage = cv2.resize(grayImageOrig, dim, interpolation=cv2.INTER_AREA)

    # Compute SIFT features for rescaled image
    kp_,des2 = sift.detector.detectAndCompute(resizedImage, None)
    kp_gray_ = cv2.drawKeypoints(resizedImage, kp_, resizedImage)
    cv2.imshow("Upscaled keypoints", kp_gray_)
    cv2.waitKey(100)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    print(len(matches))

    # Apply ratio test
    """This test rejects poor matches by computing the ratio between the best and second-best match. 
    If the ratio is below some threshold, the match is discarded as being low-quality."""
    good = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good.append([m])

    # Draw matches
    result = cv2.drawMatchesKnn(
        grayImageOrig, kp,
        resizedImage, kp_,
        good, None, flags=2)

    cv2.imshow("Matched points", result)
    cv2.waitKey(0)
