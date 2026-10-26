"""Flask app that serves the fitness activity classifier."""
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

app = Flask(__name__)

_artifact = joblib.load(MODEL_PATH)
model = _artifact["model"]
FEATURES = _artifact["features"]


def parse_features(payload: dict) -> pd.DataFrame:
    """Validate the payload and return a one-row DataFrame in training order."""
    missing = [f for f in FEATURES if f not in payload or payload[f] in ("", None)]
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")
    try:
        values = {f: float(payload[f]) for f in FEATURES}
    except (TypeError, ValueError):
        raise ValueError("All inputs must be numbers.")
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        X = parse_features(payload)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    label = model.predict(X)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(X)[0].round(3).tolist()))
    return jsonify(activity=label, probabilities=probabilities)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
