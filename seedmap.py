from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" in request.files:
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            latitude = request.form.get("latitude", "N/A")
            longitude = request.form.get("longitude", "N/A")
            return jsonify({"message": "File uploaded", "file_path": file_path, "latitude": latitude, "longitude": longitude})
    return jsonify({"error": "No file uploaded"})

if __name__ == "__main__":
    app.run(debug=True)

