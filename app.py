# app.py
import streamlit as st
import tensorflow as tf
import pydicom
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('dicom_to_jpg_model.h5')

# Streamlit app code
st.title('Your Streamlit App Title')

# Add Streamlit components (e.g., file uploader, input fields, etc.)
uploaded_file = st.file_uploader('Upload a DICOM file', type=['dcm'])

if uploaded_file is not None:
    # Preprocess the DICOM file
    dicom_data = pydicom.dcmread(uploaded_file)
    image_array = dicom_data.pixel_array.astype(np.float32)
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    input_image = normalized_image.reshape(1, *normalized_image.shape, 1)

    # Make predictions using your model
    prediction = model.predict(input_image)

    # Display the results
    st.image(normalized_image, caption='Uploaded DICOM Image', use_column_width=True)
    st.write(f'Model Prediction: {prediction[0, 0]}')
