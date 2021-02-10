"""
Main script

"""
import cv2
import numpy as np

from auto_correspondences import auto
from manual_correspondencies import manual
from homography import homography
from accuracy import accuracy

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Running main')

    #Load the data:
    img1 = cv2.imread('input_data/gingerbread1.png', -1)  # queryImage
    img2 = cv2.imread('input_data/gingerbread2.png', -1)  # trainImage
    print("Loaded images")

    #Get autocorrspondences and homography matrix
    auto_pts1, auto_pts2, auto_H = auto().keypoints()
    pre_auto, rmse_auto = accuracy().acc(auto_pts1, auto_pts2, auto_H)
    print("Autocorrespondence precision:", pre_auto)
    print("Autocorrespondence rmse:", rmse_auto)
    #Visualise homography predicted points:



    #Get manual correspondeces and homography matrix:
    man_pts1, man_pts2 = manual().keypoints()
    man_H = homography(man_pts1,man_pts2, img1, img2).get_h()  #Get homography matrix and save correspondences image
    print("manH", man_H)
    pre_manual, rmse_manual = accuracy().acc(man_pts1,man_pts2,man_H)
    print("Manual correspondence precision:", pre_manual)
    print("Manual correspondence rmse:", rmse_manual)
    #Visualise homography predicted points: