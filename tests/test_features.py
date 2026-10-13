"""Unit tests for vibration feature extraction and ISO 10816 classification."""

import unittest
import numpy as np
from src.features import (
    extract_time_domain_features,
    compute_fft_spectrum,
    classify_iso_severity,
    extract_all_vibration_features,
    ISO_10816_THRESHOLDS
)
from src.generate_sample_data import generate_vibration_signal


class TestVibrationFeatures(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 20000
        self.healthy_raw = generate_vibration_signal(duration=0.2, sample_rate=self.sample_rate, is_faulty=False, random_seed=42)
        self.faulty_raw = generate_vibration_signal(duration=0.2, sample_rate=self.sample_rate, is_faulty=True, random_seed=42)
        self.healthy_signal = self.healthy_raw[:, 0]
        self.faulty_signal = self.faulty_raw[:, 0]

    def test_extract_time_domain_features_keys(self):
        feats = extract_time_domain_features(self.healthy_signal)
        expected_keys = [
            "peak", "p2p", "rms", "mean", "variance", "std",
            "crest_factor", "kurtosis", "skewness", "shape_factor",
            "impulse_factor", "clearance_factor"
        ]
        for key in expected_keys:
            self.assertIn(key, feats)
            self.assertIsInstance(feats[key], float)

    def test_kurtosis_and_crest_higher_in_faulty(self):
        healthy_feats = extract_time_domain_features(self.healthy_signal)
        faulty_feats = extract_time_domain_features(self.faulty_signal)

        # Faulty bearing signal contains transient shockwaves -> higher kurtosis & peak
        self.assertGreater(faulty_feats["kurtosis"], healthy_feats["kurtosis"])
        self.assertGreater(faulty_feats["peak"], healthy_feats["peak"])
        self.assertGreater(faulty_feats["rms"], healthy_feats["rms"])

    def test_empty_signal_raises_error(self):
        with self.assertRaises(ValueError):
            extract_time_domain_features(np.array([]))

    def test_compute_fft_spectrum(self):
        freqs, amps, metrics = compute_fft_spectrum(self.healthy_signal, sample_rate=self.sample_rate)
        self.assertEqual(len(freqs), len(amps))
        self.assertIn("dominant_frequency_hz", metrics)
        self.assertIn("spectral_centroid_hz", metrics)
        self.assertIn("dominant_peaks", metrics)
        self.assertEqual(len(metrics["dominant_peaks"]), 5)
        self.assertIn("energy_bands", metrics)

    def test_iso_10816_severity_zones(self):
        # Testing Class II thresholds: Good <= 1.12, Acceptable <= 2.80, Alert <= 7.10, Critical > 7.10
        res_a = classify_iso_severity(0.5, machine_class="Class II")
        self.assertEqual(res_a["zone"], "Zone A")
        self.assertEqual(res_a["status"], "Good")

        res_b = classify_iso_severity(2.0, machine_class="Class II")
        self.assertEqual(res_b["zone"], "Zone B")
        self.assertEqual(res_b["status"], "Acceptable")

        res_c = classify_iso_severity(5.0, machine_class="Class II")
        self.assertEqual(res_c["zone"], "Zone C")
        self.assertEqual(res_c["status"], "Unsatisfactory / Alert")

        res_d = classify_iso_severity(12.0, machine_class="Class II")
        self.assertEqual(res_d["zone"], "Zone D")
        self.assertEqual(res_d["status"], "Unacceptable / Critical")

    def test_extract_all_vibration_features(self):
        profile = extract_all_vibration_features(self.healthy_signal, sample_rate=self.sample_rate)
        self.assertIn("time_domain", profile)
        self.assertIn("frequency_domain", profile)
        self.assertIn("iso_severity", profile)
        self.assertIn("duration_seconds", profile)
        self.assertEqual(profile["sample_rate"], self.sample_rate)


if __name__ == "__main__":
    unittest.main()
