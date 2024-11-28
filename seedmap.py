from flask import Flask, render_template, request, jsonify
import os
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # Check if file upload is provided
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            latitude = request.form.get("latitude", "N/A")
            longitude = request.form.get("longitude", "N/A")
            return jsonify({
                "message": "File uploaded successfully",
                "file_path": "/" + file_path,
                "latitude": latitude,
                "longitude": longitude,
            })

        # Check if image data is provided (captured image)
        elif "image_data" in request.form:
            image_data = request.form["image_data"]
            image_data = image_data.split(",")[1]  # Remove the data URL header
            image_bytes = base64.b64decode(image_data)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], "captured_image.png")
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            latitude = request.form.get("latitude", "N/A")
            longitude = request.form.get("longitude", "N/A")
            return jsonify({
                "message": "Image captured successfully",
                "file_path": "/" + file_path,
                "latitude": latitude,
                "longitude": longitude,
            })

        # No valid data provided
        else:
            return jsonify({"error": "No image data or file provided"}), 400

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
