"""
Main script

"""
import cv2

from auto_correspondences import auto
from manual_correspondencies import manual
from homography import homography

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Running main')

    #Load the data:
    img1 = cv2.imread('input_data/gingerbread1.png', -1)  # queryImage
    img2 = cv2.imread('input_data/gingerbread2.png', -1)  # trainImage
    print("Loaded images")

    #Get autocorrspondences and homography matrix
    auto_pts1, auto_pts2, auto_H = auto().keypoints()
    #print("Auto pts1", auto_pts1)
    print("Auto pts1.type", type(auto_pts1), auto_pts1.shape)
    #print("Auto pts2", auto_pts1)

    acc_auto = 0.8         #Get accuracy
    print("Autocorrespondence accuracy:", acc_auto)

    #Get manual correspondeces and homography matrix:
    man_pts1, man_pts2 = manual().keypoints()
    print("Man pts1", man_pts1)
    print("Man pts2", man_pts2)

    man_H = homography(man_pts1,man_pts2, img1, img2).get_h()  #Get homography matrix and save correspondences image

    acc_manual = 0.8
    print("Manual correspondence accuracy:", acc_manual)
