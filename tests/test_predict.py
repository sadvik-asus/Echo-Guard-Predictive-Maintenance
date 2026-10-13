"""Unit tests for EchoGuard CLI batch prediction and reporting module."""

import unittest
import tempfile
import json
import csv
from pathlib import Path
import numpy as np

from src.generate_sample_data import generate_vibration_signal, save_nasa_format_file
from src.predict import (
    load_vibration_file,
    predict_from_signal,
    batch_predict,
    format_cli_table,
    _heuristic_confidence,
)


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.tmp_dir.name)

        # Create sample healthy and faulty files
        self.healthy_file = self.dir_path / "healthy.txt"
        self.faulty_file = self.dir_path / "faulty.txt"

        h_data = generate_vibration_signal(duration=0.2, is_faulty=False, random_seed=1)
        f_data = generate_vibration_signal(duration=0.2, is_faulty=True, random_seed=2)

        save_nasa_format_file(h_data, self.healthy_file)
        save_nasa_format_file(f_data, self.faulty_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_vibration_file(self):
        sig, sr = load_vibration_file(self.healthy_file, channel=0, sample_rate=20000)
        self.assertEqual(len(sig), 4000)
        self.assertEqual(sr, 20000)

    def test_predict_from_signal_healthy(self):
        sig, sr = load_vibration_file(self.healthy_file)
        diag = predict_from_signal(sig, model=None, sample_rate=sr)
        self.assertIn("prediction", diag)
        self.assertIn("confidence", diag)
        self.assertIn("system_status", diag)
        self.assertTrue(diag["is_healthy"])
        self.assertEqual(diag["prediction"], "HEALTHY")

    def test_predict_from_signal_faulty(self):
        sig, sr = load_vibration_file(self.faulty_file)
        diag = predict_from_signal(sig, model=None, sample_rate=sr)
        self.assertIn("prediction", diag)
        self.assertFalse(diag["is_healthy"])
        self.assertEqual(diag["prediction"], "FAULTY")
        self.assertEqual(diag["system_status"], "CRITICAL_ALERT")

    def test_batch_predict(self):
        files = [self.healthy_file, self.faulty_file]
        results = batch_predict(files, model=None)
        self.assertEqual(len(results), 2)
        preds = [r["prediction"] for r in results]
        self.assertIn("HEALTHY", preds)
        self.assertIn("FAULTY", preds)

    def test_format_cli_table(self):
        files = [self.healthy_file, self.faulty_file]
        results = batch_predict(files, model=None)
        table_str = format_cli_table(results)
        self.assertIn("FILENAME", table_str)
        self.assertIn("healthy.txt", table_str)
        self.assertIn("faulty.txt", table_str)

    def test_heuristic_confidence_edge_cases(self):
        normal_conf = _heuristic_confidence(kurtosis=3.0, crest=2.8, rms=0.08)
        faulty_conf = _heuristic_confidence(kurtosis=7.2, crest=6.5, rms=0.85)
        self.assertGreater(normal_conf, 0.7)
        self.assertLess(faulty_conf, 0.4)


if __name__ == "__main__":
    unittest.main()
