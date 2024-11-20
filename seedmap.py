
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from urllib.parse import unquote
import requests

app = Flask(__name__)

# Load the trained model
MODEL_PATH = './models/soil_classification_pretrained_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Soil types and crop recommendations
SOIL_TYPES = ["clay", "loamy", "sandy", "chalky", "silt"]  # Example soil types
CROP_DATA = {
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

WEATHER_API_KEY = "338334a0dcbd49acb1b3dd06d9ec26f2"
WEATHER_API_URL = "https://api.weatherbit.io/v2.0/current"


def preprocess_image(image_data):
    """
    Decodes a Base64 image, preprocesses it, and returns the ready-to-predict tensor.
    """
    try:
        # Decode the Base64 string
        decoded_image = base64.b64decode(image_data)
    except Exception as e:
        app.logger.error(f"Base64 decoding failed: {e}")
        raise ValueError("Invalid Base64 image data")
    
    try:
        # Convert the decoded bytes into an image
        image = Image.open(BytesIO(decoded_image)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize
        return np.expand_dims(image_array, axis=0)
    except UnidentifiedImageError as e:
        app.logger.error(f"Image processing failed: {e}")
        raise ValueError("Image could not be identified or processed")



def predict_soil_type(image_tensor):
    """
    Predicts the soil type using the model.
    """
    predictions = model.predict(image_tensor)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100
    soil_type = SOIL_TYPES[predicted_index]
    return soil_type, confidence


def get_weather_data(latitude, longitude):
    """
    Retrieves the current temperature using the Weatherbit API.
    """
    try:
        response = requests.get(
            WEATHER_API_URL,
            params={"lat": latitude, "lon": longitude, "key": WEATHER_API_KEY},
        )
        response.raise_for_status()
        weather_data = response.json()
        return weather_data.get("data", [{}])[0].get("app_temp")
    except (requests.RequestException, KeyError) as e:
        app.logger.error(f"Weather API error: {e}")
        raise RuntimeError("Failed to retrieve weather data")


def recommend_crops(temperature, soil_type):
    """
    Recommends crops based on temperature and soil type.
    """
    recommended = []
    for crop, data in CROP_DATA.items():
        temp_range = data["temperature_range"]
        if temp_range[0] <= temperature <= temp_range[1] and soil_type in data["soil_type"]:
            recommended.append(crop)
    return recommended


@app.route('/predict', methods=['GET'])
def predict_and_recommend():
    """
    Main endpoint for predicting soil type and recommending crops.
    """
    # Extract query parameters
    image_data = request.args.get('image_data')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    if not image_data or not latitude or not longitude:
        return jsonify({"error": "Missing required parameters: image_data, latitude, longitude"}), 400

    try:
        # Decode image data
        image_data = unquote(image_data)
        image_tensor = preprocess_image(image_data)

        # Predict soil type
        soil_type, confidence = predict_soil_type(image_tensor)

        # Get temperature from weather API
        temperature = get_weather_data(float(latitude), float(longitude))
        if temperature is None:
            raise RuntimeError("Temperature data not available")

        # Recommend crops
        crops = recommend_crops(temperature, soil_type)

        # Construct response
        response = {
            "Predicted Soil Type": soil_type,
            "Confidence": f"{confidence:.2f}%",
            "Current Temperature": temperature,
            "Recommended Crops": crops,
        }
        return jsonify(response)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500


@app.route('/')
def home():
    return jsonify({"message": "API is live"}), 200


if __name__ == '__main__':
    app.run(debug=True)

