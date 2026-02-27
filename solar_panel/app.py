from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# -------- Lazy Model Loading --------
model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model("solar_model.h5", compile=False)
    return model

# Class labels
classes = ["Clean", "Dusty", "Bird-drop", "Crack", "Snow", "Electrical-damage"]

@app.route("/")
def home():
    return "Solar Panel Model Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # Open and preprocess image
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((244, 244))  # MUST match training size

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # Load model only when needed
        model = load_model_once()

        # 🔥 Get raw predictions (logits)
        prediction = model.predict(img)

        # 🔥 Convert logits to probabilities
        probabilities = tf.nn.softmax(prediction[0]).numpy()

        class_index = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": classes[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)