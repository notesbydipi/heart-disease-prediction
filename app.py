from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

# Disable GPU + logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model("model_attention.h5")
scaler = joblib.load("scaler.pkl")
print("Model loaded successfully!")

labels = [
    "Normal",
    "Supraventricular",
    "Ventricular",
    "Fusion",
    "Unknown"
]

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        if not body or "ecg" not in body:
            return jsonify({"error": "No ECG data received"}), 400

        data = body["ecg"]

        if len(data) != 187:
            return jsonify({"error": "Please enter exactly 187 ECG values"}), 400

        ecg = np.array(data, dtype=np.float32).reshape(1, 187)

        ecg_scaled = scaler.transform(ecg)
        ecg_input = ecg_scaled.reshape(1, 187, 1)

        pred = model.predict(ecg_input, verbose=0)
        predicted_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        return jsonify({
            "class": labels[predicted_class],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, port=5000)