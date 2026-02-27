from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("solar_model.h5")

classes = ["Clean", "Dusty", "Bird-drop", "Crack", "Snow"]

@app.route("/")
def home():
    return "Solar Panel Model Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    return jsonify({
        "prediction": classes[class_index],
        "confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)