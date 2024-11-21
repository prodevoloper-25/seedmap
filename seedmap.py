from flask import Flask, request
import os

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request contains the file and the other data
    if 'file' not in request.files:
        return 'No file part', 400

    # Check for latitude and longitude in the form data
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    
    if latitude is None or longitude is None:
        return 'Latitude and longitude are required', 400
    
    # Get the image file from the request
    file = request.files['file']
    
    # Check if the file has a valid name (not empty)
    if file.filename == '':
        return 'No selected file', 400

    # Save the file to the server
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Process latitude, longitude, and the file
    # For example, just print them to confirm data receipt
    print(f"Received file: {filename}")
    print(f"Latitude: {latitude}, Longitude: {longitude}")
    
    return jsonify({"text": "File Uploaded"})

if __name__ == '__main__':
    app.run(debug=True)
