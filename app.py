"""
Flask web application for Phishing URL Detection.
"""
import os
import json
from flask import Flask, render_template, request, jsonify
from predictor import get_predictor

app = Flask(__name__)
app.secret_key = "phishing-detector-secret-2024"

predictor = get_predictor()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or request.form.get("url", "")).strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Basic sanitisation – prepend scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        result = predictor.predict(url)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Return model metadata as JSON."""
    with open(os.path.join("models", "metadata.json")) as f:
        meta = json.load(f)
    return jsonify(meta)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Random Forest"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
