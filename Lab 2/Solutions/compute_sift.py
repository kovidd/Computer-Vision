# Sample solution code for computing SIFT feature pre-lab task

import cv2

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.03
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector


if __name__ == '__main__':
    # TASK 1

    # 1. Read image
    image = cv2.imread("image1.jpg")
    cv2.imshow("Colour picture", image)

    # 2. Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Initialize SIFT detector
    sift = SiftDetector()

    # 4. Detect SIFT features
    kp = sift.detector.detect(gray, None)

    # 5. Visualize detected features
    kp_gray = cv2.drawKeypoints(image, kp, gray)

    # Print number of SIFT features detected
    print("Number of SIFT features (default settings): ", len(kp))

    cv2.imshow("Keypoints image", kp_gray)
    cv2.imwrite('Keypoints image.jpg', kp_gray)
    cv2.waitKey(0)
