import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")
emotion = ["Happy", "Sad", "Angry", "Neutral"]

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (128,128)) / 255.0
img = img.reshape(1,128,128,1)

pred = model.predict(img)
print("Detected Emotion:", emotion[np.argmax(pred)])

import matplotlib.pyplot as plt

confidence = pred[0]

plt.bar(["Happy","Sad","Angry","Neutral"], confidence)
plt.title("Emotion Prediction Confidence")
plt.ylabel("Probability")
plt.show()
