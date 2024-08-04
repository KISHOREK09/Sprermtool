from flask import Flask, render_template, request, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Ensure upload and processed folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load pre-trained model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/sperm', methods=['POST', 'GET'])
def sperm():
    if request.method == 'POST':
        if 'file' not in request.files:
            m = 'No file part'
            return render_template('sperm.html', m=m)
        file = request.files['file']
        if file.filename == '':
            m = 'No file detected'
            return render_template('sperm.html', m=m)
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Call the sperm detection function
                processed_filepath, detected = detect_sperm(filepath)
                
                if detected:
                    m = 'Sperm detected and marked in the image.'
                else:
                    m = 'No sperm detected in the image.'

                return render_template('sperm.html', m=m, image_url=url_for('static', filename=f'processed/{os.path.basename(processed_filepath)}'))
            except Exception as e:
                print(e)
                m = 'Failed to process image'
                return render_template('sperm.html', m=m)
        else:
            m = "Invalid file type"
            return render_template('sperm.html', m=m)
    return render_template('sperm.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

def detect_sperm(filepath):
    # Load and preprocess the image
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized).reshape(-1, 224, 224, 3)
    img_array = img_array / 255.0  # Rescale

    # Predict using the model
    prediction = model.predict(img_array)
    detected = prediction[0][0] > 0.5  # Threshold

    # Process the image and save
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))
    
    if detected:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply GaussianBlur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold the grayscale image
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around the contours (sperms)
        marked_img = img.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            marked_img = cv2.rectangle(marked_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle
        
        cv2.imwrite(processed_filepath, marked_img)
    else:
        cv2.imwrite(processed_filepath, img)  # Save the original image if no sperm detected

    return processed_filepath, detected

if __name__ == "__main__":
    app.run(debug=True)
