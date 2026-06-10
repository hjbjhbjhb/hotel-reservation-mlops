"""
Week 5 — Flask Prediction API
Hotel Reservation Cancellation Prediction

Run:  python src/app.py
POST: http://localhost:5000/predict
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = Flask(__name__)

# Load artefacts at startup
with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)
with open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb") as f:
    ENCODERS = pickle.load(f)

FEATURE_ORDER = [
    "no_of_adults", "no_of_children", "no_of_weekend_nights",
    "no_of_week_nights", "type_of_meal_plan", "required_car_parking_space",
    "room_type_reserved", "lead_time", "arrival_year", "arrival_month",
    "arrival_date", "market_segment_type", "repeated_guest",
    "no_of_previous_cancellations", "no_of_previous_bookings_not_canceled",
    "avg_price_per_room", "no_of_special_requests",
]

CAT_FEATURES = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]


def encode_input(data: dict) -> np.ndarray:
    row = []
    for feat in FEATURE_ORDER:
        val = data[feat]
        if feat in CAT_FEATURES:
            val = ENCODERS[feat].transform([val])[0]
        row.append(float(val))
    return np.array(row).reshape(1, -1)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X = encode_input(payload)
        X_scaled = SCALER.transform(X)
        pred = int(MODEL.predict(X_scaled)[0])
        prob = float(MODEL.predict_proba(X_scaled)[0][1])
        label = "Canceled" if pred == 1 else "Not_Canceled"
        return jsonify({
            "prediction": label,
            "cancel_probability": round(prob, 4),
            "confidence": round(max(prob, 1 - prob), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
