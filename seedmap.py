from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import requests
from geopy.geocoders import Nominatim

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Define the same training directory to get class indices
train_dir = './data/train'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
class_indices = train_generator.class_indices
soil_types = list(class_indices.keys())

# Define crop data
crop_data = {
    "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
    "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
    "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
    # Add remaining crops as per earlier code
}

# Temporary storage for uploaded files
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess and predict soil type
def predict_soil_type(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return soil_types[predicted_class], predictions[0][predicted_class] * 100

# Function to fetch weather data
def get_weather(lat, lon):
    parameters = {
        "lat": lat,
        "lon": lon,
        "key": "338334a0dcbd49acb1b3dd06d9ec26f2"
    }
    response = requests.get("https://api.weatherbit.io/v2.0/current", params=parameters)
    data = response.json()
    temp = data["data"][0]["app_temp"]
    return temp

# Find suitable crops
def find_suitable_crops(temperature, soil_type):
    suitable_crops = []
    for crop, requirements in crop_data.items():
        temp_range = requirements["temperature_range"]
        soil_types = requirements["soil_type"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
            suitable_crops.append(crop)
    return suitable_crops

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded file
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get geolocation data
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        
        if not lat or not lon:
            return "Latitude and Longitude not available", 400
        
        # Predict soil type and get weather data
        soil_type, confidence = predict_soil_type(file_path)
        temperature = get_weather(lat, lon)
        recommended_crops = find_suitable_crops(temperature, soil_type)

        return render_template('index.html', 
                               soil_type=soil_type, 
                               confidence=confidence,
                               temperature=temperature, 
                               recommended_crops=recommended_crops)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
