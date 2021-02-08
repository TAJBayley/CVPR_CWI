"""
Script to find keypoint correspondences
both manually and automatically
Linnea Evanson
07/02/21

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


class auto():
    #Automatic keypoint detection
    def __init__(self, input_data = 0):
        self.input_data = input_data

    def keypoints(self):
        print("Finding auto keypoints")

        MIN_MATCH_COUNT = 10

        img1 = cv2.imread('input_data/gingerbread1.png', -1)  # queryImage
        img2 = cv2.imread('input_data/gingerbread2.png', -1)  # trainImage
        print("Loaded images")

        # Initiate SIFT detector
        #sift = cv2.SIFT()
        sift = cv2.SIFT_create()
        print("Initiated SIFT")

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        print("Found Keypoints")
        print("len kp1", len(kp1))
        print("len kp2", len(kp2))
        print("kp1", kp1)
        print("kp2", kp2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        print("matches size", len(matches))

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        print("good", good)
        print("good length", len(good))

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, _ = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        else:
            print
            "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            matchesMask = None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        print("Saving autocorrespondence image")
        #plt.imshow(img3, 'gray') #, plt.show()
        plt.imsave("output_images/auto_correspondences.png",img3)

        return src_pts, dst_pts, M

#
# test = auto()
# matches = test.keypoints()
# print(matches)