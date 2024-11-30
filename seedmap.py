from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/upload', methods=['POST'])
def upload_file():
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file provided!"}), 400

    image = Image.open(file.stream)
    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    soil_types = ["clay", "silt", "loamy", "sandy"]
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    response = {
        "soil_type": soil_types[predicted_class],
        "confidence": confidence * 100,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": 25,  # Mock temperature for now
        "suitable_crops": ["Wheat", "Rice"]  # Mock crop data for now
    }

    return jsonify(response)
