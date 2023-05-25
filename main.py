import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

def main():
    image = cv2.imread("data/inputs/06_test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = filters.unsharp_mask(gray)
    plt.imshow(gray, cmap='gray')
    plt.show()
    # sato = filters.sato(gray)
    # vessels = sato > 0.012
    # plt.imshow(vessels, cmap='gray')
    # plt.show()
    #
    # mask = cv2.imread("data/masks/06_test_mask.jpg")
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = mask > 100
    #
    # vessels[mask == 0] = 0
    # plt.imshow(vessels, cmap='gray')
    # plt.show()
    #
    # white = (255, 255, 255)
    # blue = (0, 0, 255)
    # red = (255, 0, 0)
    # green = (0, 255, 0)
    #
    # conf_matrix = np.zeros(image.shape)
    #
    # true_positive = false_positive = false_negative = true_negative = 0
    #
    # shape = vessels.shape
    # width, height = shape[0], shape[1]
    #
    # model = cv2.imread("data/models/06_test_model.jpg")
    # model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
    # model = model > 10
    # plt.imshow(model, cmap='gray')
    # plt.show()
    #
    # for x in range(width):
    #     for y in range(height):
    #
    #         if vessels[x][y] and model[x][y]:
    #             conf_matrix[x][y] = green
    #             true_positive += 1
    #
    #         elif vessels[x][y] and not model[x][y]:
    #             conf_matrix[x][y] = red
    #             false_positive += 1
    #
    #         elif not vessels[x][y] and model[x][y]:
    #             conf_matrix[x][y] = blue
    #             false_negative += 1
    #
    #         elif not vessels[x][y] and not model[x][y]:
    #             conf_matrix[x][y] = white
    #             true_negative += 1
    # plt.imshow(conf_matrix)
    # plt.show()
    #
    # accuracy = round((true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative), 4)
    # sensitivity = round(true_positive / (true_positive + false_negative + 1), 4)
    # specificity = round(true_negative / (false_positive + true_negative + 1), 4)
    # precision = round(true_positive / (true_positive + false_positive + 1), 4)
    # g_mean = round(math.sqrt(sensitivity * specificity), 4)
    # f_measure = round((2 * precision * sensitivity) / (precision + sensitivity + 1), 4)
    #
    # print("Accuracy: ", accuracy)
    # print("Sensitivity: ", sensitivity)
    # print("Specificity: ", specificity)
    # print("Precision: ", precision)
    # print("G-mean: ", g_mean)
    # print("F-measure: ", f_measure)
    #
    #
    patch_size = 5
    features = []
    height, width = gray.shape

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            patch = gray[y:y+patch_size, x:x+patch_size]

            # Compute desired features from the patch
            variance = np.var(patch)
            moments = cv2.moments(patch)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Concatenate the features into a single feature vector
            feature_vector = np.hstack([hu_moments, variance])
            features.append(feature_vector)
    print(features)


if __name__ == '__main__':
    main()


