"""Unit tests for EchoGuard Industrial IoT REST API module."""

import unittest
import io
import json
from src.api import create_app
from src.generate_sample_data import generate_vibration_signal


class TestApi(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("service", data)
        self.assertIn("endpoints", data)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)

    def test_analyze_with_json_signal(self):
        sig = generate_vibration_signal(duration=0.1, sample_rate=20000, is_faulty=False, random_seed=42)[:, 0]
        payload = {
            "signal": sig.tolist(),
            "sample_rate": 20000,
            "machine_class": "Class II"
        }
        response = self.client.post("/api/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("diagnosis", data)
        self.assertIn("advisory", data)
        self.assertEqual(data["diagnosis"]["prediction"], "HEALTHY")
        self.assertIn("key_metrics", data["diagnosis"])

    def test_analyze_missing_signal_returns_400(self):
        response = self.client.post("/api/analyze", json={"sample_rate": 20000})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_analyze_with_file_upload(self):
        sig = generate_vibration_signal(duration=0.1, sample_rate=20000, is_faulty=True, random_seed=123)[:, 0]
        txt_content = "\n".join(f"{val:.6f}\t0.0\t0.0\t0.0" for val in sig)

        data = {
            "file": (io.BytesIO(txt_content.encode("utf-8")), "sensor_reading.txt")
        }
        response = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        res_data = response.get_json()
        self.assertIn("diagnosis", res_data)
        self.assertIn("advisory", res_data)

    def test_advisor_endpoint(self):
        payload = {
            "prediction": "FAULTY",
            "confidence": 0.2,
            "features": {
                "time_domain": {"kurtosis": 5.5, "crest_factor": 4.8, "rms": 0.2},
                "frequency_domain": {"dominant_frequency_hz": 236.4}
            }
        }
        response = self.client.post("/api/advisor", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["severity_level"], "CRITICAL")
        self.assertIn("sop_action_plan", data)

    def test_kb_search_endpoint(self):
        response = self.client.get("/api/kb/search?q=lubrication&limit=2")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["query"], "lubrication")
        self.assertGreater(data["count"], 0)


if __name__ == "__main__":
    unittest.main()
