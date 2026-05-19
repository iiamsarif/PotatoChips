from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "potato_model.keras")

# Load model once at startup
model = load_model(MODEL_PATH, compile=False)

# Keep the same order as training
class_names = ["Early", "Healty", "Late"]

IMG_SIZE = (224, 224)


def get_recommendation(label):
    label_lower = label.lower()

    if "early" in label_lower:
        return {
            "title": "Early Blight Detected",
            "advice": [
                "Remove infected leaves",
                "Avoid overhead watering",
                "Use fungicide if needed",
            ],
        }
    elif "late" in label_lower:
        return {
            "title": "Late Blight Detected",
            "advice": [
                "Improve air circulation",
                "Remove infected plants",
                "Apply fungicide as recommended",
            ],
        }
    else:
        return {
            "title": "Healthy Leaf",
            "advice": [
                "Plant looks healthy",
                "Continue regular care",
                "Monitor plants weekly",
            ],
        }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Preprocess image
        img = load_img(file, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction) * 100)

        recommendation = get_recommendation(predicted_class)

        probabilities = [
            {
                "label": class_names[i],
                "probability": float(prediction[i] * 100),
            }
            for i in range(len(class_names))
        ]

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
