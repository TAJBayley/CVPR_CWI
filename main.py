"""
Main script

"""
import cv2
import numpy as np

from auto_correspondences import auto
from manual_correspondencies import manual
from homography import homography
from accuracy import accuracy
from calibration import Calibration
from resize import resize

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Running main')

    # # Initialise calibration object
    # image = cv2.imread('input_data/fd_6.JPG')
    # test = Calibration(img_size=resize(image, 30).shape)
    # # Get corners for all images
    # for i in range(6, 11):
    #     image = cv2.imread('input_data/fd_{}.JPG'.format(i))
    #     image = resize(image, 30)
    #     test.getCorners(image)
    #
    # # Get Intrinsic and Extrinsic matrices:
    # K, R, t = test.parameters()

    #Load the data:
    img1 = cv2.imread('input_data/hg_1.JPG')  # queryImage
    img1 = resize(img1, 30)
    img2 = cv2.imread('input_data/hg_2.JPG')  # trainImage
    img2 = resize(img2, 30)

    print("Loaded images")

    #Get autocorrspondences and homography matrix
    auto_pts1, auto_pts2, auto_H = auto(img1,img2).keypoints()
    pre_auto, rmse_auto = accuracy().acc(auto_pts1, auto_pts2, auto_H)
    print("Autocorrespondence precision:", pre_auto)
    print("Autocorrespondence rmse:", rmse_auto)
    #Visualise homography predicted points:



    #Get manual correspondeces and homography matrix:
    man_pts1, man_pts2 = manual(img1,img2).keypoints()
    man_H = homography(man_pts1,man_pts2, img1, img2).get_h()  #Get homography matrix and save correspondences image
    print("manH", man_H)
    pre_manual, rmse_manual = accuracy().acc(man_pts1,man_pts2,man_H)
    print("Manual correspondence precision:", pre_manual)
    print("Manual correspondence rmse:", rmse_manual)
    #Visualise homography predicted points:
