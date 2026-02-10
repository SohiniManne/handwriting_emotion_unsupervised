# ✍️ Handwriting Emotion Recognition System

### Machine Learning–Based (Semi-Supervised)

## 📌 Project Overview

Handwritten text reflects not only the content written but also the emotional state of the writer through variations in stroke, pressure, spacing, and slant. Manual analysis of handwriting emotions is subjective and inconsistent.
This project implements a **machine learning–based handwriting emotion recognition system** that automatically analyzes handwriting patterns and identifies the emotional state of the writer.

The system uses a **semi-supervised learning approach**, enabling emotion recognition **without manual labeling** of handwriting images.

---

## 🎯 Objectives

* Analyze handwriting style instead of text content
* Automatically extract handwriting features
* Discover emotion-related handwriting patterns
* Generate emotion labels without manual intervention
* Train a CNN model to classify emotions
* Provide an interactive dashboard for emotion prediction

---

## 🧠 Emotions Detected

* Happy
* Sad
* Angry
* Neutral

---

## 📂 Dataset

* **Dataset Used:** IAM Handwriting Top50 (Kaggle)
* **Total Samples:** 4,899 handwritten images
* **Labels:** Not provided (handled using semi-supervised learning)

---

## ⚙️ System Architecture

```
Handwriting Image
        ↓
Feature Extraction
        ↓
Unsupervised Clustering
        ↓
Automatic Emotion Label Generation
        ↓
CNN Training
        ↓
Emotion Prediction
        ↓
Dashboard Visualization
```

---

## 🧪 Technologies Used

* Python 3.10
* OpenCV
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Streamlit

---

## 📁 Project Structure

```
handwriting_emotion_unsupervised/
 ├── data/
 │    └── iam/                  # Handwriting images
 ├── extract_features.py
 ├── generate_labels.py
 ├── train_cnn.py
 ├── predict.py
 ├── visualize_clusters.py
 ├── visualize_features.py
 ├── app.py                     # Dashboard
 ├── features.npy
 ├── clusters.npy
 ├── pseudo_labels.npy
 ├── model.h5
 └── README.md
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install opencv-python numpy matplotlib scikit-learn tensorflow streamlit pillow
```

---

### 2️⃣ Feature Extraction

```bash
python extract_features.py
```

Creates: `features.npy`

---

### 3️⃣ Generate Emotion Labels

```bash
python generate_labels.py
```

Creates:

* `clusters.npy`
* `pseudo_labels.npy`

---

### 4️⃣ Train CNN Model

```bash
python train_cnn.py
```

Creates: `model.h5`

---

### 5️⃣ Predict Emotion (Single Image)

Place a handwriting image in the project folder as `test.png`, then run:

```bash
python predict.py
```

---

### 6️⃣ Run Dashboard

```bash
streamlit run app.py
```

Opens the dashboard in your browser for interactive emotion prediction.

---

## 📊 Visualizations Included

* PCA-based cluster visualization
* Feature distribution histograms
* CNN accuracy and loss graphs
* Emotion prediction confidence bar chart
* Interactive dashboard using Streamlit

---

## ✅ Key Features

* Fully automated (no manual labeling)
* Semi-supervised learning approach
* Robust handling of corrupted images
* Real-time emotion prediction
* User-friendly dashboard

---

## ⚠️ Limitations

* Emotion labels are automatically inferred
* Emotional interpretation is not clinically validated
* Writing style may vary across individuals

---

## 🚀 Future Enhancements

* Add more handwriting features (slant, baseline detection)
* Improve emotion mapping strategies
* Extend to real-time handwriting input
* Deploy as a web application

---

## 🏁 Conclusion

This project demonstrates that handwriting style contains emotion-related information and that machine learning can effectively analyze and identify emotional states from handwriting. The proposed system successfully executes the problem statement and provides a scalable foundation for future research in handwriting-based emotion analysis.

---

## 👨‍💻 Author

**[Sohini Manne]**
Handwriting Emotion Recognition System
Academic Project

Just tell me 👍
