from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Path to the trained model
MODEL_PATH =r"D:\tumordatas\brain_tumor_model.h5"  # Ensure this file exists in the project folder
model = tf.keras.models.load_model(MODEL_PATH)

# Define the tumor classes
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Preprocess image to fit the model's input size"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))  # Resize to fit model
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route("/")
def home():
    """Render the HTML frontend"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image uploads and return predictions"""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Preprocess image
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]  # Get top class
    confidence = round(float(np.max(prediction) * 100), 2)  # Confidence %

    return jsonify({"label": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
