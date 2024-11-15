from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# Dummy data generator to get class indices
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

# Function to preprocess and predict soil type
def predict_soil_type(base64_image):
    # Decode the base64 image
    image_data = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the soil type
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class] * 100
    return soil_types[predicted_class], confidence

# Route for predicting soil type
@app.route('/predict_soil/<base_var>', methods=['POST'])
def predict():
    # Get JSON data
    # data = request.get_json()
    # if 'image' not in data:
    #     return jsonify({"error": "No image data provided"}), 400

    base64_image = base_var

    try:
        # Perform prediction
        soil_type, confidence = predict_soil_type(base64_image)
        result = {
            "Predicted Soil Type": soil_type,
            "Confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return the result as JSON
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
