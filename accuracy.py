"""
Calculate accuracy from homography or fundamental matrix
Linnea Evanson
09/02/21

"""
import numpy as np

class accuracy():
    def acc(self, auto_pts1, auto_pts2, auto_H):
        auto_pts1 = np.squeeze(auto_pts1)
        auto_pts1 = np.append(auto_pts1, [[1] for i in range(auto_pts1.shape[0])], axis = 1) #reshape, add coord of 1

        auto_predicted_pts2 = np.zeros((auto_pts1.shape[0], auto_pts1.shape[1], 1))
        for row in range(len(auto_pts1)):
            coord = np.expand_dims(auto_pts1[row], axis = 1)   #add coord of 1 to z dimension, so can multiply by 3x3 h matrix
            auto_predicted_pts2[row] = np.matmul(auto_H , coord) #matrix multiply
            auto_predicted_pts2[row] = auto_predicted_pts2[row] / auto_predicted_pts2[row][2] #divide by z' to get real image coords

        print("Is z coord 1?", auto_predicted_pts2[:4])

        #Accuracy is calculated in terms of precision
        auto_pts2 = np.squeeze(auto_pts2)
        auto_pts2 = np.append(auto_pts2, [[1] for i in range(auto_pts2.shape[0])], axis=1)  # reshape, add coord of 1

        precision = np.mean( (auto_pts2 - np.squeeze(auto_predicted_pts2)) / auto_pts2)
        rmse =  np.sqrt(np.mean( (auto_pts2 - np.squeeze(auto_predicted_pts2)) **2)  )

        return precision, rmse

    # def visualise(self):
