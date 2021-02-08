"""
Script to find manual correspondences
both manually and automatically
Linnea Evanson
07/02/21

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

class manual():
    #Automatic keypoint detection
    def __init__(self, input_data = 0):
        self.input_data = input_data
        self.refPt1 = []
        self.refPt2 = []
        self.num_keypoints = 0

    def keypoints(self):
        print("Finding keypoints")

        # load the image, clone it, and setup the mouse callback function
        self.image = cv2.imread('input_data/gingerbread1.png')
        self.image2 = cv2.imread('input_data/gingerbread2.png')

        cv2.namedWindow("IMAGE1")
        cv2.setMouseCallback("IMAGE1", self.clickImage1)   #not sure what this does?????

        cv2.namedWindow("IMAGE2")
        cv2.setMouseCallback("IMAGE2", self.clickImage2)  # not sure what this does?????

        # keep looping until the 'q' key is pressed
        while True:
            # display the FIRST image and wait for a keypress
            cv2.imshow("IMAGE1", self.image)
            cv2.imshow("IMAGE2", self.image2)

            key = cv2.waitKey(1) & 0xFF

            # if the 'c' key is pressed, break from the loop
            if key == ord("c"):
                break

        # close all open windows
        cv2.destroyAllWindows()
        #Convert to np arrays:
        keypoints1 = np.asarray(self.refPt1)
        keypoints2 = np.asarray(self.refPt2)

        return keypoints1, keypoints2


    def clickImage1(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt1.append([[x, y]])
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            self.num_keypoints += 1
            print("Number of Keypoints:",self.num_keypoints)


    def clickImage2(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt2.append([(x, y)])
            cv2.circle(self.image2, (x, y), 5, (0, 255, 0), -1)

# test = manual()
# kpts1, kpts2 = test.keypoints()
# print("Number of Keypoints found:", len(kpts1))
# for i in range(len(kpts1)):
#     print(kpts1[i],"   ",kpts2[i])