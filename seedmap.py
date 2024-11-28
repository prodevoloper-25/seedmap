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

# Load the trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Define the class indices (adjust based on your training classes)
class_indices = {0: "clay", 1: "loamy", 2: "sandy", 3: "chalky", 4: "silt"}
crop_data = {
    "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
    "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
    "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
    # Add more crops here
}

def predict_soil_type(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return class_indices[predicted_class], confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Handle image file
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
        # Handle base64 image data
        elif "image_data" in request.form:
            image_data = request.form["image_data"].split(",")[1]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "captured_image.png")
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(image_data))
        else:
            return jsonify({"error": "No image data provided"}), 400

        # Process location
        latitude = request.form.get("latitude", "N/A")
        longitude = request.form.get("longitude", "N/A")

        # Predict soil type
        soil_type, confidence = predict_soil_type(file_path)

        # Fetch temperature from weather API
        response = requests.get("https://api.weatherbit.io/v2.0/current", params={
            "lat": latitude,
            "lon": longitude,
            "key": "338334a0dcbd49acb1b3dd06d9ec26f2"
        })
        data = response.json()
        temp = data["data"][0]["app_temp"]

        # Find suitable crops
        suitable_crops = [
            crop for crop, info in crop_data.items()
            if info["temperature_range"][0] <= temp <= info["temperature_range"][1] and soil_type in info["soil_type"]
        ]

        return jsonify({
            "message": "Processing successful",
            "soil_type": soil_type,
            "confidence": confidence,
            "temperature": temp,
            "suitable_crops": suitable_crops,
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

