# 🖼️ VGG19 Image Classification Web App

A web-based image classification application built with TensorFlow/Keras using the pretrained VGG19 convolutional neural network model. The app allows users to upload an image or capture one directly from their webcam, and it classifies the image into one of the predefined ImageNet categories.

This project demonstrates how a deep learning model can be deployed interactively through an intuitive Streamlit web interface.

---

## 🚀 Features

- 📸 Upload or capture images directly via webcam  
- 🧠 Uses pretrained VGG19 (ImageNet) model from TensorFlow/Keras  
- 🔍 Automatically resizes all images to 224×224 pixels  
- ⚙️ Preprocesses input and returns predicted class & confidence score  
- ⚡ Streamlit caching for faster performance (avoids model reloads)  
- 💡 Clean and responsive web interface

---

## 🏗 Project Architecture

```
User Upload / Webcam Capture
            │
            ▼
Image Preprocessing (resize to 224x224, normalize)
            │
            ▼
VGG19 Model (pretrained on ImageNet)
            │
            ▼
Prediction (Top Class + Confidence)
            │
            ▼
Display Results in Streamlit Interface
```
---

## 🧠 Model Overview

The VGG19 model is a 19-layer deep convolutional neural network trained on over 1 million images from the ImageNet dataset.  
It can recognize 1000 object categories, such as animals, vehicles, or everyday items.

---

## 📦 Dependencies

The app uses the following Python libraries:
```
streamlit==1.55.0
tensorflow==2.21.0
numpy==2.2.6
pandas==2.3.3
matplotlib==3.10.8
```
Install them using:

```bash
pip install -r requirements.txt
```
---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run classification.py
```
Once running, open the local address shown in the terminal (usually):

```bash
http://localhost:8501
```
From there, you can upload an image or take a photo via webcam and view the classification result.
 
---

## 🌐 Deployment

This app is hosted on Streamlit Cloud, accessible directly from the browser — no installation required.

---

## 🔮 Possible Future Improvements
```
🧱 Add support for other pretrained models (ResNet, MobileNet, etc.)
⚙️ Include top-k predictions visualization
📁 Allow custom-trained model upload
🎨 Enhance UI with richer image result displays
```
---
## 🛠 Tech Stack
```
Python
TensorFlow / Keras
Streamlit
NumPy
Pandas
Matplotlib
``` 
---

## ⚠️ Disclaimer
This project uses a pretrained model for educational and demonstration purposes.
Predictions may not always be accurate and should not be used for high-stakes or production-critical applications.

---

## 👤 Creator
SRIJAN KUMAR SAHA

---

## 📜 License

This project is intended for educational and research purposes.