
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
import base64
from io import BytesIO
from PIL import Image
from urllib.parse import unquote

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Define the training directory to get class indices
train_dir = './data/train'
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Class indices for soil types
class_indices = train_generator.class_indices
soil_types = list(class_indices.keys())

# Crop data dictionary
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

# Function to preprocess and predict soil type
def predict_soil_type(image_data):
    try:
        # Decode and preprocess the image
        decoded_image = base64.b64decode(image_data)
        image = Image.open(BytesIO(decoded_image)).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Model prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        soil_type = soil_types[predicted_class]
        return soil_type, confidence
    except (base64.binascii.Error, Image.UnidentifiedImageError) as e:
        app.logger.error(f"Error decoding or processing image: {str(e)}")
        raise ValueError("Invalid image data")

# Function to find suitable crops
def find_suitable_crops(temperature, soil_type):
    suitable_crops = []
    for crop, requirements in crop_data.items():
        temp_range = requirements["temperature_range"]
        soil_types = requirements["soil_type"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
            suitable_crops.append(crop)
    return suitable_crops

# Route to predict soil type and recommend crops
@app.route('/predict', methods=['GET'])
def predict_and_recommend():
    # Retrieve request parameters
    image_data = request.args.get('image_data')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    # Validate inputs
    if not image_data or not latitude or not longitude:
        return jsonify({"error": "Missing required parameters: image_data, latitude, and longitude"}), 400

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except ValueError:
        return jsonify({"error": "Invalid latitude or longitude values"}), 400

    try:
        # Decode the image data (handle URL encoding)
        image_data = unquote(image_data)
        soil_type, confidence = predict_soil_type(image_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Get the weather data
    weather_api_key = '338334a0dcbd49acb1b3dd06d9ec26f2'
    try:
        weather_response = requests.get(
            "https://api.weatherbit.io/v2.0/current",
            params={"lat": latitude, "lon": longitude, "key": weather_api_key}
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        temperature = weather_data.get("data", [{}])[0].get("app_temp")
    except (requests.RequestException, KeyError) as e:
        app.logger.error(f"Weather API error: {str(e)}")
        return jsonify({"error": "Failed to retrieve weather data"}), 500

    if temperature is None:
        return jsonify({"error": "Could not retrieve temperature data"}), 500

    # Find suitable crops
    recommended_crops = find_suitable_crops(temperature, soil_type)

    # Return the result
    result = {
        "Predicted Soil Type": soil_type,
        "Confidence": f"{confidence:.2f}%",
        "Current Temperature": temperature,
        "Recommended Crops": recommended_crops
    }
    return jsonify(result)

# Optional root route for testing
@app.route('/')
def home():
    return jsonify({"message": "API is live and running"}), 200

if __name__ == '__main__':
    app.run(debug=True)
