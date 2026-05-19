from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import tempfile

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "potato_model.keras"
model = load_model(MODEL_PATH)

# Keep same class order as training
class_names = ["Early", "Healty", "Late"]

IMG_SIZE = (224, 224)

def get_recommendation(label: str):
    label_lower = label.lower()

    if "early" in label_lower:
        return {
            "title": "Early Blight Detected",
            "advice": [
                "Remove infected leaves",
                "Avoid overhead watering",
                "Use fungicide if needed"
            ]
        }
    elif "late" in label_lower:
        return {
            "title": "Late Blight Detected",
            "advice": [
                "Improve air circulation",
                "Remove infected plants",
                "Apply fungicide as recommended"
            ]
        }
    else:
        return {
            "title": "Healthy Leaf",
            "advice": [
                "Plant looks healthy",
                "Continue regular care",
                "Monitor plants weekly"
            ]
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
        return jsonify({"error": "Empty filename"}), 400

    # Save temporarily
    suffix = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        # Load and preprocess image
        img = load_img(temp_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction) * 100)

        recommendation = get_recommendation(predicted_class)

        probabilities = [
            {"label": class_names[i], "probability": float(prediction[i] * 100)}
            for i in range(len(class_names))
        ]

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "recommendation": recommendation,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)