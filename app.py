import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image

# Ensure model exists
MODEL_PATH = "models/terrorist_detector.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Function to make predictions
def predict_image(image):
    try:
        # Convert image to NumPy array
        image = Image.open(image)
        image_np = np.array(image)

        # Ensure correct format (RGB)
        if image_np.shape[-1] == 4:  # If RGBA, convert to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Convert to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Resize image
        image_resized = cv2.resize(image_bgr, (224, 224))

        # Preprocess image for MobileNetV2
        image_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)
        image_preprocessed = np.expand_dims(image_preprocessed, axis=0)

        # Make prediction
        prediction = model.predict(image_preprocessed)[0][0]

        return "🟥 Terrorist" if prediction > 0.5 else "🟩 Not a Terrorist"
    except Exception as e:
        return f"⚠️ Error processing image: {str(e)}"

# Streamlit UI
st.title("🕵️‍♂️ Terrorist Detector")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Predict and display result
    result = predict_image(uploaded_file)
    st.subheader(f"🔍 Prediction: {result}")
