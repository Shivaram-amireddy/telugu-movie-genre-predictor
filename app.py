import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("telugu_genre_model.h5")

import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load dataset just to get class_names correctly
dataset = image_dataset_from_directory("C:/Users/shiva/posters_dataset", image_size=(224, 224), batch_size=32)
class_names = dataset.class_names


# Genre class names based on your dataset folder names
class_names = ['action', 'comedy', 'drama', 'horror', 'romance']

# App title
st.title("🎬 Telugu Movie Poster Genre Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a movie poster image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))  # Make sure this matches your training size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)

        # Check for valid class index
        if predicted_index < len(class_names):
            predicted_class = class_names[predicted_index]
            st.image(img, caption='Uploaded Poster', use_column_width=True)
            st.success(f"🎯 Predicted Genre: **{predicted_class.upper()}**")
        else:
            st.error("Prediction index is out of range. Please check your class names.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
