from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)

IMG_SIZE = (224, 224)
class_names = ["Early", "Healty", "Late"]

def build_model():
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = build_model()
model.load_weights("potato_weights.weights.h5")
print("Model loaded successfully")

def recommend(label):
    label = label.lower()
    if "early" in label:
        return ["Remove infected leaves", "Avoid overhead watering", "Use fungicide if needed"]
    if "late" in label:
        return ["Improve air circulation", "Remove infected plants", "Apply fungicide as recommended"]
    return ["Plant looks healthy", "Continue regular care", "Monitor weekly"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = load_img(file, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(pred))
    label = class_names[idx]
    confidence = float(np.max(pred) * 100)

    return jsonify({
        "predicted_class": label,
        "confidence": round(confidence, 2),
        "probabilities": [
            {"label": class_names[i], "probability": float(pred[i] * 100)}
            for i in range(3)
        ],
        "recommendation": {
            "title": label,
            "advice": recommend(label)
        }
    })
