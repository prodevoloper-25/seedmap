from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Set up a directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if an image file was uploaded
    if 'file' not in request.files:
        return jsonify(error="No file part in the request"), 400
    
    file = request.files['file']
    
    # Check for latitude and longitude
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    
    if not latitude or not longitude:
        return jsonify(error="Latitude and longitude are required"), 400

    # Verify the file has a filename
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    # Respond with success and uploaded data confirmation
    return jsonify(success="File successfully uploaded", 
                   filename=file.filename, 
                   latitude=latitude, 
                   longitude=longitude), 200

if __name__ == '__main__':
    app.run(debug=True)
