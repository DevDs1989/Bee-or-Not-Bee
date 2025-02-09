import os
import pickle

import cv2
from skimage.feature import hog

model_path = "model.pkl"
encoder_path = "encoder.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)
with open(encoder_path, "rb") as file:
    label_encoder = pickle.load(file)


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} not found")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features, hog_image = hog(
        gray_image,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        channel_axis=None,
    )
    return hog_features


def predict_image(image_path, model, label_encoder):
    features = preprocess_image(image_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class


while True:
    image_path = input("\nEnter the path to your image (or 'q' to quit): ")
    if image_path.lower() == "q":
        break

    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
        continue
    try:
        predicted_class = predict_image(image_path, model, label_encoder)
        print(f"\nPredicted class: {predicted_class}")

    except Exception as e:
        print(f"Error: {e}")
 