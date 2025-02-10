import imghdr
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features, hog_image = hog(
        gray_image,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None,
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_features, hog_image_rescaled

def show_hog_image(index):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, labels[index], os.listdir(os.path.join(DATA_DIR, labels[index]))[index])), cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("HOG Image")
    plt.imshow(hog_images[index], cmap='gray')

    plt.show()


DATA_DIR = "./data"
labels = []
images = []
hog_images = []
image_extensions = ["jpg", "jpeg", "png"]
for dir in os.listdir(DATA_DIR):
    label = dir
    for file in os.listdir(os.path.join(DATA_DIR, dir)):
        image_path = os.path.join(DATA_DIR, dir, file)
        try:
            image = cv2.imread(image_path)
            if imghdr.what(image_path) in image_extensions:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (64, 64))
                hog_features,hog_image = extract_hog_features(image)
                images.append(hog_features)
                labels.append(label)
                hog_images.append(hog_image)
            else:
                print(f"Invalid image extension: {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Invalid image: {image_path}")
            os.remove(image_path)
            continue


X = np.array(images)
Y = np.array(labels)
print(f"Loaded {len(X)} images with shape {X[0].shape} and {len(Y)} labels")

print(f"X: {X.shape}, Y: {Y.shape}")
le = LabelEncoder()
Y = le.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)} images, Test: {len(X_test)} images")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# show_hog_image(12)

# sample_size, img_size = X_train.shape
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

svm = SVC(kernel="linear", random_state=42)

svm.fit(X_train_flatten, Y_train)
Y_pred = svm.predict(X_test_flatten)

print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
print(f"Classification Report: {classification_report(Y_test, Y_pred)}")

with open("model.pkl", "wb") as model:
    pickle.dump(svm, model)
with open("encoder.pkl", "wb") as encoder:
    pickle.dump(le, encoder)
print("Model and encoder saved successfully")
