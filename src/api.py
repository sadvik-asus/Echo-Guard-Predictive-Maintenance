"""
EchoGuard Industrial IoT REST API Service
Provides RESTful endpoints for edge sensor telemetry ingestion,
real-time CNN predictive maintenance inference, ISO 10816 severity scoring,
and prescriptive maintenance advisory generation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

from src.config import (
    DEFAULT_MODEL_PATH,
    SAMPLE_RATE
)
from src.features import extract_all_vibration_features
from src.predict import predict_from_signal, load_vibration_file
from src.advisor import generate_maintenance_advisory, query_knowledge_base

try:
    import tensorflow as tf
    from models.model import load_echo_guard_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def create_app(model_path: str = str(DEFAULT_MODEL_PATH)) -> Flask:
    """
    Application factory for EchoGuard REST API.
    """
    app = Flask(__name__)
    if CORS_AVAILABLE:
        CORS(app)

    # Initialize model
    app.config["MODEL"] = None
    app.config["MODEL_STATUS"] = "No pre-trained model file loaded (running physics heuristic mode)"

    if TF_AVAILABLE and Path(model_path).exists():
        loaded = load_echo_guard_model(model_path)
        if loaded is not None:
            app.config["MODEL"] = loaded
            app.config["MODEL_STATUS"] = f"Loaded CNN weights from {Path(model_path).name}"

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "service": "EchoGuard Industrial Predictive Maintenance API",
            "version": "1.0.0",
            "status": "operational",
            "model_engine": app.config["MODEL_STATUS"],
            "endpoints": {
                "GET /api/health": "Service health check and engine info",
                "POST /api/analyze": "Analyze raw vibration signal array or uploaded sensor file",
                "POST /api/advisor": "Generate prescriptive maintenance advisory and SOP",
                "GET /api/kb/search": "Search ISO maintenance knowledge base"
            }
        }), 200

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "tensorflow_available": TF_AVAILABLE,
            "model_loaded": app.config["MODEL"] is not None,
            "engine_status": app.config["MODEL_STATUS"]
        }), 200

    @app.route("/api/analyze", methods=["POST"])
    def analyze_vibration():
        """
        Accepts either:
          1. JSON payload: {"signal": [0.01, ...], "sample_rate": 20000, "machine_class": "Class II"}
          2. Multipart form data with file upload under key 'file'.
        """
        sample_rate = request.args.get("sample_rate", default=SAMPLE_RATE, type=int)
        machine_class = request.args.get("machine_class", default="Class II", type=str)

        # Scenario 1: Uploaded File
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file.filename:
                return jsonify({"error": "Empty filename provided"}), 400

            suffix = Path(uploaded_file.filename).suffix
            # Save temporarily for loader
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                uploaded_file.save(tmp.name)
                tmp_path = tmp.name

            try:
                sig, sr = load_vibration_file(Path(tmp_path), channel=0, sample_rate=sample_rate)
            except Exception as e:
                return jsonify({"error": f"Failed to parse uploaded sensor file: {e}"}), 400
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Scenario 2: JSON Signal Payload
        elif request.is_json:
            data = request.get_json()
            if not data or "signal" not in data:
                return jsonify({"error": "Missing 'signal' array in JSON body"}), 400

            raw_sig = data.get("signal")
            if not isinstance(raw_sig, list) or len(raw_sig) == 0:
                return jsonify({"error": "'signal' must be a non-empty list of numerical samples"}), 400

            sig = np.array(raw_sig, dtype=float)
            sample_rate = int(data.get("sample_rate", sample_rate))
            machine_class = str(data.get("machine_class", machine_class))

        else:
            return jsonify({"error": "Request must contain multipart 'file' or JSON body with 'signal'"}), 400

        # Run diagnosis
        diagnosis = predict_from_signal(
            signal=sig,
            model=app.config["MODEL"],
            sample_rate=sample_rate,
            machine_class=machine_class
        )

        # Generate accompanying advisory
        advisory = generate_maintenance_advisory(
            prediction=diagnosis["prediction"],
            confidence=diagnosis["confidence"],
            features=diagnosis["full_features"]
        )

        response = {
            "diagnosis": diagnosis,
            "advisory": advisory
        }
        return jsonify(response), 200

    @app.route("/api/advisor", methods=["POST"])
    def get_advisory():
        """
        Generates maintenance SOP report for an arbitrary prediction state.
        Payload: {"prediction": "FAULTY", "confidence": 0.25, "features": {...}}
        """
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        body = request.get_json()
        prediction = body.get("prediction", "FAULTY")
        confidence = float(body.get("confidence", 0.5))
        features = body.get("features", None)

        advisory = generate_maintenance_advisory(
            prediction=prediction,
            confidence=confidence,
            features=features
        )
        return jsonify(advisory), 200

    @app.route("/api/kb/search", methods=["GET"])
    def search_kb():
        query = request.args.get("q", default="", type=str)
        top_k = request.args.get("limit", default=3, type=int)

        results = query_knowledge_base(query, top_k=top_k)
        return jsonify({
            "query": query,
            "count": len(results),
            "results": results
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGuard REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to .keras model")
    args = parser.parse_args()

    custom_app = create_app(model_path=args.model_path)
    print(f"🔊 Starting EchoGuard REST API on http://{args.host}:{args.port}")
    custom_app.run(host=args.host, port=args.port, debug=False)
