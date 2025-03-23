import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("models/terrorist_detector.h5")

def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)[0][0]
    return "Terrorist" if prediction > 0.5 else "Not a Terrorist"

# Test with an image
image_path = "test.jpg"
result = predict_image(image_path)
print(f"Prediction: {result}")
