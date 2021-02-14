"""
Script for 3D epipolar lines, stereorectification and depth map.
Linnea Evanson
13/02/21

"""
from __future__ import print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt
from resize import resize

class threeD():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2


    def depth_map(self):
        imgL = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        #1.  Set of hyperparams
        stereo = cv2.StereoBM_create(numDisparities= 16, blockSize=15) #numDisparities is window size, must be divisible by 16

        disparity = stereo.compute(imgL,imgR)
        plt.imshow(disparity,'gray')
        plt.show()

        #2.  Another set of Hyperparams:--------------------------
        # imgL = self.img1
        # imgR = self.img2
        # window_size = 3
        # min_disp = 16
        # num_disp = 112 - min_disp
        # stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
        #                               numDisparities=num_disp,
        #                               blockSize=16,
        #                               P1=8 * 3 * window_size ** 2,
        #                               P2=32 * 3 * window_size ** 2,
        #                               disp12MaxDiff=1,
        #                               uniquenessRatio=10,
        #                               speckleWindowSize=100,
        #                               speckleRange=32
        #                               )
        #
        # print('computing disparity...')
        # disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        #
        #
        # imgL = resize(imgL,30) #resize to visualise on screen
        # cv2.imshow('left', imgL)
        # disp_img = (disp - min_disp) / num_disp
        # disp_img = resize(disp_img,30)
        # cv2.imshow( 'disparity', disp_img)
        # cv2.waitKey()
        #
        # print('Done')

        #3.  Another set of hyperparams:----------------
        # imgL = cv2.resize(imgL, (600, 600))
        # imgR = cv2.resize(imgR, (600, 600))
        #
        # # Setting parameters for StereoSGBM algorithm
        # minDisparity = 0
        # numDisparities = 64
        # blockSize = 8
        # disp12MaxDiff = 1
        # uniquenessRatio = 10
        # speckleWindowSize = 10
        # speckleRange = 8
        #
        # # Creating an object of StereoSGBM algorithm
        # stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
        #                                numDisparities=numDisparities,
        #                                blockSize=blockSize,
        #                                disp12MaxDiff=disp12MaxDiff,
        #                                uniquenessRatio=uniquenessRatio,
        #                                speckleWindowSize=speckleWindowSize,
        #                                speckleRange=speckleRange
        #                                )
        #
        # # Calculating disparith using the StereoSGBM algorithm
        # disp = stereo.compute(imgL, imgR).astype(np.float32)
        #
        # # Calculating disparith using the StereoSGBM algorithm
        # disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
        #
        # # Displaying the disparity map
        # cv2.imshow("disparity", disp)
        # cv2.imshow("left image", imgL)
        # cv2.imshow("right image", imgR)
        # cv2.waitKey(0)

        #4.   Another set of hyperparams -----------------------------

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        # block_size = 11
        # min_disp = -128
        # max_disp = 128
        # # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # # In the current implementation, this parameter must be divisible by 16.
        # num_disp = max_disp - min_disp
        # # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # # Normally, a value within the 5-15 range is good enough
        # uniquenessRatio = 5
        # # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        # speckleWindowSize = 200
        # # Maximum disparity variation within each connected component.
        # # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # # Normally, 1 or 2 is good enough.
        # speckleRange = 2
        # disp12MaxDiff = 0
        #
        # stereo = cv2.StereoSGBM_create(
        #     minDisparity=min_disp,
        #     numDisparities=num_disp,
        #     blockSize=block_size,
        #     uniquenessRatio=uniquenessRatio,
        #     speckleWindowSize=speckleWindowSize,
        #     speckleRange=speckleRange,
        #     disp12MaxDiff=disp12MaxDiff,
        #     P1=8 * 1 * block_size * block_size,
        #     P2=32 * 1 * block_size * block_size,
        # )
        # disparity_SGBM = stereo.compute(imgL, imgR)
        #
        # # Normalize the values to a range from 0..255 for a grayscale image
        # disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
        #                               beta=0, norm_type=cv2.NORM_MINMAX)
        # disparity_SGBM = np.uint8(disparity_SGBM)
        # cv2.imshow("Disparity", disparity_SGBM)
        # cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM)

        #5.  ---------------

        num_disp = 128;
        block_size = 25;
        speckleRange = 4;
        speckleWindowSize = 200;


        # elseif
        # true
        # bm = cv.StereoBM();
        # bm.NumDisparities = 192;
        # bm.BlockSize = 21;
        # else
        # bm = cv.StereoSGBM('MinDisparity', -64, 'NumDisparities', 192, ...
        # 'BlockSize', 11, 'P1', 100, 'P2', 1000, 'Disp12MaxDiff', 32, ...
        # 'PreFilterCap', 0, 'UniquenessRatio', 15, ...
        # 'SpeckleWindowSize', 1000, 'SpeckleRange', 16, 'Mode', 'HH');

        stereo = cv2.StereoSGBM_create(
            #minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            #uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            #disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        disparity_SGBM = stereo.compute(imgL, imgR)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv2.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)
        cv2.imshow("Disparity", disparity_SGBM)
        cv2.imwrite("disparity_SGBM_norm_1.png", disparity_SGBM)
