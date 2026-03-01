from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model("solar_models.h5", compile=False)
    return model

# ⚠ MUST MATCH TRAINING ORDER EXACTLY
classes = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]

@app.route("/")
def home():
    return "Solar Panel Fault Detection Model Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Open and resize image (same size as training)
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((244, 244))

        # Convert to numpy
        img = np.array(image)
        img = np.expand_dims(img, axis=0)

        # 🔥 IMPORTANT: Use VGG16 preprocessing (NOT /255)
        img = preprocess_input(img)

        # Load model
        model = load_model_once()

        # Predict (softmax already inside model)
        prediction = model.predict(img)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": classes[class_index],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)