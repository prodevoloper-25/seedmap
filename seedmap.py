from flask import Flask, render_template, request, jsonify
import os
import base64
import tensorflow as tf
import numpy as np
import cv2
import requests

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the pre-trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Dummy data generator to get class indices (adjust the path to your dataset)
train_dir = './data/train'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define crop recommendations
crop_data = {
    "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
    "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
    "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
    "Sugarcane": {"temperature_range": (20, 30), "soil_type": ["loamy", "clay"]},
    "Potato": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy"]},
    "Cotton": {"temperature_range": (25, 35), "soil_type": ["sandy", "loamy"]},
    # Add other crops here
}

# Predict soil type function
def predict_soil_type(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    class_indices = train_generator.class_indices
    soil_types = list(class_indices.keys())
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return soil_types[predicted_class], confidence

# Find suitable crops function
def find_suitable_crops(temperature, soil_type):
    suitable_crops = []
    for crop, requirements in crop_data.items():
        temp_range = requirements["temperature_range"]
        soil_types = requirements["soil_type"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
            suitable_crops.append(crop)
    return suitable_crops

# API route to upload image and get crop recommendation
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Handle image upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            # Predict soil type
            soil_type, confidence = predict_soil_type(file_path)

            # Fetch temperature using weather API
            latitude = request.form.get("latitude", "0.0")
            longitude = request.form.get("longitude", "0.0")
            weather_api_key = "YOUR_WEATHERBIT_API_KEY"  # Replace with your API key
            response = requests.get(
                f"https://api.weatherbit.io/v2.0/current?lat={latitude}&lon={longitude}&key={weather_api_key}"
            )
            temp = response.json()["data"][0]["app_temp"]

            # Recommend suitable crops
            recommended_crops = find_suitable_crops(temp, soil_type)
            return jsonify({
                "soil_type": soil_type,
                "confidence": confidence,
                "temperature": temp,
                "recommended_crops": recommended_crops
            })

        else:
            return jsonify({"error": "No image file provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
