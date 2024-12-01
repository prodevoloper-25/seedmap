from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = './models/soil_classification_pretrained_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Data directory to get class indices
TRAIN_DIR = './data/train'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
soil_classes = list(train_generator.class_indices.keys())

# Crop recommendations
crop_data = {
    "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
    "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
    "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
    "Potato": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy"]},
    "Cotton": {"temperature_range": (25, 35), "soil_type": ["sandy", "loamy"]},
}

# Function to predict soil type
def predict_soil_type(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = soil_classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class, confidence

# Function to fetch weather data
def get_weather(lat, lon):
    api_key = "d062fd060c9c4a8abd5c6b2e006f08cd"
    url = f"https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={api_key}"
    response = requests.get(url)
    weather_data = response.json()
    return weather_data["data"][0]["app_temp"]

# Function to find suitable crops
def find_suitable_crops(temperature, soil_type):
    suitable_crops = []
    for crop, requirements in crop_data.items():
        temp_range = requirements["temperature_range"]
        soil_types = requirements["soil_type"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
            suitable_crops.append(crop)
    return suitable_crops

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle uploaded file
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get latitude and longitude
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        if not latitude or not longitude:
            return jsonify({"error": "Latitude or longitude not provided"}), 400

        # Predict soil type
        soil_type, confidence = predict_soil_type(file_path)

        # Get weather data
        temperature = get_weather(latitude, longitude)

        # Find suitable crops
        recommended_crops = find_suitable_crops(temperature, soil_type)

        # Render results
        return render_template(
            "index.html",
            soil_type=soil_type,
            confidence=confidence,
            temperature=temperature,
            recommended_crops=recommended_crops,
            show_results=True,
        )

    return render_template("index.html", show_results=False)

if __name__ == "__main__":
    app.run(debug=True)

