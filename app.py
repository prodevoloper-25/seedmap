from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Define the same training directory to get class indices
train_dir = './data/train'

# Dummy data generator to get class indices
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


# Function to preprocess and predict soil type
def predict_soil_type(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the soil type
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return soil_types[predicted_class], confidence


# Route for predicting soil type
@app.route('/predict_soil', methods=['POST'])
def predict():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file to a temporary location
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    try:
        # Perform prediction
        soil_type, confidence = predict_soil_type(file_path)
        result = {
            "Predicted Soil Type": soil_type,
            "Confidence": f"{confidence:.2f}%"
        }
    finally:
        # Clean up: remove the temporary file after prediction
        os.remove(file_path)

    # Return the result as JSON
    return jsonify(result)


# Run the app
if __name__ == '__main__':
    # Ensure the 'temp' directory exists
    
    app.run(debug=True)
