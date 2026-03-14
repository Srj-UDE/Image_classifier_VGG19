import streamlit as st
import time
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from PIL import Image

@st.cache_resource
def load_model():
    model = vgg19.VGG19()
    return model

model = load_model()

st.header("Image Classifier with VGG19", divider="rainbow")

st.sidebar.markdown("### A Project by SRIJAN KUMAR SAHA")

st.markdown("This project is a web-based image classification app that allows users to upload images or capture them directly from the webcam. The app uses the pretrained VGG19 convolutional neural network from TensorFlow/Keras to classify images into one of the categories it was trained on.")

st.markdown("##### Project Features:\n"
    "- Accepts images via upload or webcam capture.\n"
    "- Automatically resizes images to 224x224 pixels.\n"
    "- Preprocesses images and feeds them into VGG19 for prediction.\n"
    "- Displays the predicted class and confidence in a clean, user-friendly interface.\n"
    "- Uses Streamlit caching to avoid reloading the model multiple times for faster performance.")
st.markdown("##### Take / upload a photo")

def fig_show(img,col_name): 
    fig, ax = plt.subplots()
    ax.imshow(img)
    col_name.pyplot(fig)
    

def predict(img,col_name):
    img = tf.keras.applications.vgg19.preprocess_input(img)
    #pred = model.predict(img)
    pred = model.predict(img.reshape(1,224,224,3))
    top_five_predict =vgg19.decode_predictions(pred, top=3)
    col_name.write(f"""##### Object detected: \"{top_five_predict[0][0][1]}\"""")
    if top_five_predict[0][0][2] < 0.1:
        col_name.write(f"""##### Confidence Score: {top_five_predict[0][0][2]*1000:.2f} %""")
    else:
        col_name.write(f"""##### Confidence Score: {top_five_predict[0][0][2]*100:.2f} %""")

col1 , col2 = st.columns([1,1])

uploaded_photo = col1.file_uploader("Upload a photo")
if uploaded_photo:
    col1.success("Photo uploaded successfully!")
    col1.markdown("#### Your Image")
    img = tf.keras.utils.load_img(uploaded_photo, target_size=(224,224,3))
    img= np.array(img)
    fig_show(img,col1)
    predict(img,col1)



camera_photo = col2.camera_input("Take a photo")
if camera_photo:
    col2.success("Photo taken successfully!")
    col2.markdown("#### Your Image")
    uncroped_img = Image.open(camera_photo)

    # Crop to square (center crop)
    width, height = uncroped_img.size
    min_dim = 224
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_square = uncroped_img.crop((left, top, right, bottom))
    #img = tf.keras.utils.load_img(img_square, target_size=(224,224,3))
    img= np.array(img_square)
    fig_show(img,col2)
    predict(img,col2)


def allot_photo(photo):
    if photo == uploaded_photo:
        col1.success("Photo uploaded successfully")
    elif photo == camera_photo:
        col2.success("Photo taken successfully")
    else:
        st.error("Error: Photo not found!")

