from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = r"D:\aiml project\hackthon\uploads"
MODEL_PATH = r"D:\aiml project\hackthon\runs\detect\train11\weights\best.pt"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO(MODEL_PATH)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'before' not in request.files or 'after' not in request.files:
        return "Missing files", 400

    before_img = request.files['before']
    after_img = request.files['after']

    if before_img.filename == '' or after_img.filename == '':
        return "No file selected", 400

    # Save images
    before_path = os.path.join(app.config['UPLOAD_FOLDER'], before_img.filename)
    after_path = os.path.join(app.config['UPLOAD_FOLDER'], after_img.filename)
    before_img.save(before_path)
    after_img.save(after_path)

    # Perform YOLO detection
    results_before = model.predict(before_path)
    results_after = model.predict(after_path)

    # Check if deforestation occurred (compare detections)
    deforestation_detected = len(results_before[0].boxes) < len(results_after[0].boxes)

    return render_template("result.html", deforestation=deforestation_detected)

if __name__ == '__main__':
    app.run(debug=True)
