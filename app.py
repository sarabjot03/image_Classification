import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('image_classification_model.h5')  # Save your trained model as 'model.h5'

# Define class labels
class_labels = ['Cat', 'Dog']  # Adjust based on your classes

# Streamlit app title
st.title("Image Classification: Cat vs Dog")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    img_array = np.array(image.resize((64, 64)))  # Resize to match the model input
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[int(prediction[0] > 0.5)]
    
    # Display the result
    st.write(f"Prediction: **{predicted_class}**")