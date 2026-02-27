from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("solar_model.h5", compile=False)

classes = ["Clean", "Dusty", "Bird-drop", "Crack", "Snow"]

@app.route("/")
def home():
    return "Solar Panel Model Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Open image properly
        image = Image.open(file.stream).convert("RGB")

        # Resize to correct training size
        image = image.resize((244, 244))

        # Convert to numpy
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        return jsonify({
            "prediction": classes[class_index],
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)