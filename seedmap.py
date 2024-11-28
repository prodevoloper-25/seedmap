from flask import Flask, render_template, request, jsonify
import os
import base64

app = Flask(__name__)

# Configure upload folder and maximum content length (10 MB)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # If file upload is provided
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            return process_image(file_path)

        # If base64 image data is provided
        elif "image_data" in request.form:
            image_data = request.form["image_data"]
            image_data = image_data.split(",")[1]  # Remove the data URL header
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "captured_image.png")
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            return process_image(file_path)

        # No valid data provided
        else:
            return jsonify({"error": "No image data or file provided"}), 400

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def process_image(file_path):
    # Add your soil type prediction logic here
    # For demonstration purposes, we'll return a mock result
    soil_type = "loamy"
    confidence = 95.0
    temperature = 28  # Example temperature value
    suitable_crops = ["Wheat", "Tomato"]

    return jsonify({
        "message": "Image processed successfully",
        "file_path": "/" + file_path,
        "soil_type": soil_type,
        "confidence": confidence,
        "temperature": temperature,
        "suitable_crops": suitable_crops,
    })

if __name__ == "__main__":
    app.run(debug=True)

