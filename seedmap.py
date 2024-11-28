import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Dummy data generator to get class indices
train_dir = '../data/train'
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Function to preprocess and predict soil type
def predict_soil_type(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the soil type
    predictions = model.predict(image)
    class_indices = train_generator.class_indices
    soil_types = list(class_indices.keys())
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return soil_types[predicted_class], confidence

# Function to get temperature using latitude and longitude
def get_temperature(lat, lon):
    parameters = {
        "lat": lat,
        "lon": lon,
        "key": "338334a0dcbd49acb1b3dd06d9ec26f2",
    }
    response = requests.get("https://api.weatherbit.io/v2.0/current", params=parameters)
    data = response.json()
    return data["data"][0]["app_temp"]

# Crop data
crop_data = {
    "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
    "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
    "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
    "Sugarcane": {"temperature_range": (20, 30), "soil_type": ["loamy", "clay"]},
    "Potato": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy"]},
    "Cotton": {"temperature_range": (25, 35), "soil_type": ["sandy", "loamy"]},
    "Barley": {"temperature_range": (12, 25), "soil_type": ["loamy", "chalky"]},
    "Soybean": {"temperature_range": (20, 30), "soil_type": ["loamy", "clay", "sandy"]},
    "Tomato": {"temperature_range": (18, 27), "soil_type": ["loamy", "chalky", "sandy"]},
    "Peanut": {"temperature_range": (22, 32), "soil_type": ["sandy", "chalky"]},
    "Onion": {"temperature_range": (13, 24), "soil_type": ["loamy", "sandy", "chalky"]},
    "Garlic": {"temperature_range": (10, 23), "soil_type": ["chalky", "loamy"]},
    "Carrot": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy", "silt"]},
    "Cabbage": {"temperature_range": (15, 22), "soil_type": ["clay", "silt"]},
    "Lettuce": {"temperature_range": (16, 21), "soil_type": ["silt", "loamy", "chalky"]},
    "Chili Pepper": {"temperature_range": (20, 30), "soil_type": ["loamy", "sandy"]},
}

# Function to find suitable crops based on temperature and soil type
def find_suitable_crops(temperature, soil_type):
    suitable_crops = []
    for crop, requirements in crop_data.items():
        temp_range = requirements["temperature_range"]
        soil_types = requirements["soil_type"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
            suitable_crops.append(crop)
    return suitable_crops

# Route to handle file upload or captured image
@app.route('/upload', methods=['POST'])
def upload_image():
    lat = request.form.get('latitude')
    lon = request.form.get('longitude')
    file = request.files['file']
    image = Image.open(file.stream)
    image = np.array(image)

    soil_type, confidence = predict_soil_type(image)
    temperature = get_temperature(lat, lon)
    suitable_crops = find_suitable_crops(temperature, soil_type)

    # Prepare response data
    result = {
        "soil_type": soil_type,
        "confidence": confidence,
        "temperature": temperature,
        "suitable_crops": suitable_crops,
        "latitude": lat,
        "longitude": lon
    }

    return jsonify(result)

# Home route to serve the HTML
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
