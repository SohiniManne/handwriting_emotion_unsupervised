import cv2
import os
import numpy as np

DATA_PATH = "data/iam"
IMG_SIZE = 128
features = []
image_paths = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    mean_intensity = np.mean(img)                      # pressure proxy
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (IMG_SIZE**2)   # stroke density
    texture = cv2.Laplacian(img, cv2.CV_64F).var()     # roughness

    features.append([mean_intensity, edge_density, texture])
    image_paths.append(path)

np.save("features.npy", np.array(features))
np.save("image_paths.npy", np.array(image_paths))

print("✔ Features extracted:", len(features))
