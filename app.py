import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("model.h5")
emotion_labels = ["Happy", "Sad", "Angry", "Neutral"]

st.set_page_config(page_title="Handwriting Emotion Recognition", layout="centered")

st.title("✍️ Handwriting Emotion Recognition Dashboard")
st.write("Upload a handwritten image to detect the writer's emotional state.")

uploaded_file = st.file_uploader("Upload Handwriting Image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = img.reshape(1, 128, 128, 1)

    # Predict
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]

    st.subheader(f"🧠 Detected Emotion: **{emotion}**")

    # Confidence bar chart
    st.subheader("Prediction Confidence")
    fig, ax = plt.subplots()
    ax.bar(emotion_labels, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
