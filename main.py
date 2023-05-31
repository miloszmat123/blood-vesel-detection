import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler

PATCH_SIZE = 5


def get_features(image, gray, mask):
    features = []
    height, width = gray.shape

    for y in range(0, height - PATCH_SIZE + 1):
        for x in range(0, width - PATCH_SIZE + 1):
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patch_gray = gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            mean = np.mean(patch.reshape(-1, 3), axis=0)
            std = np.std(patch.reshape(-1, 3), axis=0)
            moments = cv2.moments(patch_gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            feature_vector = np.hstack([mean, std, [x for x in moments.values()], hu_moments])
            features.append(feature_vector)
    return features


def get_labels(image, mask):
    labels = []
    height, width = image.shape

    for y in range(0, height - PATCH_SIZE + 1):
        for x in range(0, width - PATCH_SIZE + 1):
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            label = patch[PATCH_SIZE // 2, PATCH_SIZE // 2]
            labels.append(label)
    return labels


def get_predicted_image(image, predicted, mask):
    height, width = image.shape
    new_image = np.zeros((height, width))
    for y in range(0, height - PATCH_SIZE + 1):
        for x in range(0, width - PATCH_SIZE + 1):
            new_image[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = predicted[0]
            predicted = predicted[1:]
    return new_image


def train_model():
    train_set = [
        "01",
        "02",
        "03",
        "04",
        "05"
    ]
    features = []
    labels = []

    for train_image in train_set:
        image = cv2.imread("train/inputs/" + train_image + "_test.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = filters.unsharp_mask(gray)

        mask = cv2.imread("train/masks/" + train_image + "_test_mask.jpg")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask > 100
        gray[mask == 0] = 0

        model = cv2.imread("train/models/" + train_image + "_test_model.jpg")
        model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
        model = model > 10
        model[mask == 0] = 0
        features.extend(get_features(image, gray, mask))
        labels.extend(get_labels(model, mask))
    features = np.array(features)
    labels = np.array(labels)
    sampler = RandomUnderSampler(sampling_strategy=1)
    features, labels = sampler.fit_resample(features, labels)
    classifier = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    classifier.fit(features, labels)
    dump(classifier, "model.joblib")
    return classifier


def load_model():
    return load("model.joblib")


def calculate_metrics(predicted, model):
    width, height = predicted.shape
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for x in range(width):
        for y in range(height):
            if predicted[x][y] and model[x][y]:
                true_positive += 1

            elif predicted[x][y] and not model[x][y]:
                false_positive += 1

            elif not predicted[x][y] and model[x][y]:
                false_negative += 1

            elif not predicted[x][y] and not model[x][y]:
                true_negative += 1

    accuracy = round(
        (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative), 4)
    sensitivity = round(true_positive / (true_positive + false_negative + 1), 4)
    specificity = round(true_negative / (false_positive + true_negative + 1), 4)
    precision = round(true_positive / (true_positive + false_positive + 1), 4)
    g_mean = round(math.sqrt(sensitivity * specificity), 4)
    f_measure = round((2 * precision * sensitivity) / (precision + sensitivity + 1), 4)
    print("Accuracy: ", accuracy)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity)
    print("Precision: ", precision)
    print("G-mean: ", g_mean)
    print("F-measure: ", f_measure)


def calculate_confusion_matrix(image, predicted, model):
    width, height = predicted.shape
    white = (255, 255, 255)
    blue = (0, 0, 255)
    red = (255, 0, 0)
    green = (0, 255, 0)
    conf_matrix = np.zeros(image.shape)
    for x in range(width):
        for y in range(height):

            if predicted[x][y] and model[x][y]:
                conf_matrix[x][y] = green

            elif predicted[x][y] and not model[x][y]:
                conf_matrix[x][y] = red

            elif not predicted[x][y] and model[x][y]:
                conf_matrix[x][y] = blue

            elif not predicted[x][y] and not model[x][y]:
                conf_matrix[x][y] = white
    return conf_matrix


def main():
    image = cv2.imread("data/inputs/07_test.jpg")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = filters.unsharp_mask(gray)

    mask = cv2.imread("data/masks/07_test_mask.jpg")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask > 100
    gray[mask == 0] = 0

    model = cv2.imread("data/models/07_test_model.jpg")
    model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)
    model = model > 10
    model[mask == 0] = 0

    sato = filters.sato(gray)
    predicted_vision = sato > 0.012
    predicted_vision[mask == 0] = 0
    plt.imshow(predicted_vision, cmap='gray')
    plt.show()
    conf_matrix = calculate_confusion_matrix(image, predicted_vision, model)
    plt.imshow(conf_matrix)
    plt.show()
    calculate_metrics(predicted_vision, model)

    # classifier = train_model()
    classifier = load_model()
    predictions = classifier.predict(get_features(image, gray, mask))
    predicted_ml = get_predicted_image(gray, predictions, mask)
    plt.imshow(predicted_ml, cmap='gray')
    plt.show()
    conf_matrix = calculate_confusion_matrix(image, predicted_ml, model)
    plt.imshow(conf_matrix)
    plt.show()
    calculate_metrics(predicted_ml, model)


if __name__ == '__main__':
    main()
