from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import requests
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('./models/soil_classification_pretrained_model.h5')

# # Define the training directory to get class indices
# train_dir = './data/train'
# train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Class indices for soil types
# class_indices = train_generator.class_indices
# soil_types = list(class_indices.keys())

# # Crop data dictionary
# crop_data = {
#     "Wheat": {"temperature_range": (10, 25), "soil_type": ["clay", "loamy", "chalky"]},
#     "Rice": {"temperature_range": (20, 35), "soil_type": ["clay", "silt"]},
#     "Maize": {"temperature_range": (18, 27), "soil_type": ["loamy", "sandy", "silt"]},
#     "Sugarcane": {"temperature_range": (20, 30), "soil_type": ["loamy", "clay"]},
#     "Potato": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy"]},
#     "Cotton": {"temperature_range": (25, 35), "soil_type": ["sandy", "loamy"]},
#     "Barley": {"temperature_range": (12, 25), "soil_type": ["loamy", "chalky"]},
#     "Soybean": {"temperature_range": (20, 30), "soil_type": ["loamy", "clay", "sandy"]},
#     "Tomato": {"temperature_range": (18, 27), "soil_type": ["loamy", "chalky", "sandy"]},
#     "Peanut": {"temperature_range": (22, 32), "soil_type": ["sandy", "chalky"]},
#     "Onion": {"temperature_range": (13, 24), "soil_type": ["loamy", "sandy", "chalky"]},
#     "Garlic": {"temperature_range": (10, 23), "soil_type": ["chalky", "loamy"]},
#     "Carrot": {"temperature_range": (15, 20), "soil_type": ["sandy", "loamy", "silt"]},
#     "Cabbage": {"temperature_range": (15, 22), "soil_type": ["clay", "silt"]},
#     "Lettuce": {"temperature_range": (16, 21), "soil_type": ["silt", "loamy", "chalky"]},
#     "Chili Pepper": {"temperature_range": (20, 30), "soil_type": ["loamy", "sandy"]},
# }

# # Function to preprocess and predict soil type
# def predict_soil_type(image_data):
#     image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)

#     predictions = model.predict(image)
#     predicted_class = np.argmax(predictions)
#     confidence = predictions[0][predicted_class] * 100
#     soil_type = soil_types[predicted_class]
#     return soil_type, confidence

# # Function to find suitable crops
# def find_suitable_crops(temperature, soil_type):
#     suitable_crops = []
#     for crop, requirements in crop_data.items():
#         temp_range = requirements["temperature_range"]
#         soil_types = requirements["soil_type"]
#         if temp_range[0] <= temperature <= temp_range[1] and soil_type in soil_types:
#             suitable_crops.append(crop)
#     return suitable_crops

# # Route to predict soil type and recommend crops
# @app.route('/predict_and_recommend', methods=['POST'])
# def predict_and_recommend():
#     data = request.get_json()

#     # Check if image data is provided
#     if 'image' not in data or 'lat' not in data or 'lon' not in data:
#         return jsonify({"error": "Image, latitude, and longitude are required."}), 400

#     image_data = data['image']
#     latitude = data['lat']
#     longitude = data['lon']

#     # Get the weather data
#     weather_api_key = '338334a0dcbd49acb1b3dd06d9ec26f2'
#     weather_response = requests.get("https://api.weatherbit.io/v2.0/current", params={
#         "lat": latitude,
#         "lon": longitude,
#         "key": weather_api_key
#     })
#     weather_data = weather_response.json()
#     temperature = weather_data["data"][0]["app_temp"]

#     # Predict soil type
#     soil_type, confidence = predict_soil_type(image_data)

#     # Find suitable crops
#     recommended_crops = find_suitable_crops(temperature, soil_type)

#     # Return the result
#     result = {
#         "Predicted Soil Type": soil_type,
#         "Confidence": f"{confidence:.2f}%",
#         "Current Temperature": temperature,
#         "Recommended Crops": recommended_crops
#     }
#     return jsonify(result)

@app.route('/sum',methods=['GET'])
def hello():
   num1 = float(request.args.get('num1', 0))
   num2 = float(request.args.get('num2', 0))
   num3 = float(request.args.get('num3', 0))
   d = int(num1+num2+num3)
   return jsonify({"sum":d})



if __name__ == '__main__':
    app.run(debug=True)
