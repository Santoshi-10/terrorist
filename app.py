import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# Ensure model exists
MODEL_PATH = "models/terrorist_detector.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Function to make predictions
def predict_image(image):
    try:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)[0][0]
        return "Terrorist" if prediction > 0.5 else "Not a Terrorist"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Streamlit UI
st.title("🕵️‍♂️ Terrorist Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    result = predict_image(uploaded_file)
    st.subheader(f"Prediction: {result}")
