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
from fundamental import Fundamental
from resize import resize

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Running main')

    # # Initialise calibration object
    # image = cv2.imread('input_data/fd_6.JPG')
    # test = Calibration(img_size=resize(image, 30).shape)
    # # Get corners for all images
    # for i in range(8, 10):
    #     image = cv2.imread('input_data/fd_{}.JPG'.format(i))
    #     image = resize(image, 30)
    #     test.getCorners(image)
    #
    # # Get Intrinsic and Extrinsic matrices:
    # K, R, t = test.parameters()
    #
    # image = cv2.imread('input_data/fd_8.JPG')
    # image = resize(image, 30)
    # undistorted = test.undist(image)
    # print("Saving undistorted image")
    # cv2.imwrite("output_images/undistorted.png", undistorted)
    # cv2.imshow('original', image)
    # cv2.imshow('undistorted', undistorted)
    # # press any key to destroy window and continue
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Load the data:
    # img1 = cv2.imread('input_data/hg_1.JPG')  # queryImage
    # img1 = resize(img1, 30)
    # img2 = cv2.imread('input_data/hg_2.JPG')  # trainImage
    # img2 = resize(img2, 30)
    #
    # print("Loaded images")
    #
    # # Get autocorrspondences and homography matrix
    # auto_pts1, auto_pts2, auto_H = auto(img1, img2).keypoints()
    # pre_auto, rmse_auto = accuracy().acc(auto_pts1, auto_pts2, auto_H)
    # print("Autocorrespondence precision:", pre_auto)
    # print("Autocorrespondence rmse:", rmse_auto)
    # # Visualise homography predicted points:
    #
    # # Get manual correspondeces and homography matrix:
    # man_pts1, man_pts2 = manual(img1, img2).keypoints()
    # man_H = homography(man_pts1, man_pts2, img1, img2).get_h()  # Get homography matrix and save correspondences image
    # print("manH", man_H)
    # pre_manual, rmse_manual = accuracy().acc(man_pts1, man_pts2, man_H)
    # print("Manual correspondence precision:", pre_manual)
    # print("Manual correspondence rmse:", rmse_manual)
    # # Visualise homography predicted points:

    # Load the data:
    img1 = cv2.imread('input_data/hg_1.JPG')  # queryImage
    img1 = resize(img1, 30)
    img2 = cv2.imread('input_data/hg_2.JPG')  # trainImage
    img2 = resize(img2, 30)

    print("Loaded images")

    # Get fundamental matrix from correspondences
    fund = Fundamental(img1, img2, method='auto')
    fund.vanishingLines()
    # F = fund.getFundamental(algo=cv2.FM_RANSAC)
    # print('F',F)
    # fund.epilines(index=1)
    # fund.epilines(index=2)
    # cv2.imshow('img1', fund.img1)
    # cv2.imshow('img2', fund.img2)
    # cv2.imwrite('output_images/auto_epipolar_RANSAC_1.png',fund.img1)
    # cv2.imwrite('output_images/auto_epipolar_RANSAC_2.png',fund.img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
