"""Unit tests for synthetic vibration signal generation."""

import unittest
import tempfile
from pathlib import Path
import numpy as np
from src.generate_sample_data import generate_vibration_signal, save_nasa_format_file


class TestSyntheticData(unittest.TestCase):
    def test_generate_vibration_signal_dimensions(self):
        sig = generate_vibration_signal(duration=0.5, sample_rate=20000, is_faulty=False, random_seed=42)
        self.assertIsInstance(sig, np.ndarray)
        self.assertEqual(sig.shape, (10000, 4))

    def test_faulty_signal_higher_energy(self):
        healthy = generate_vibration_signal(duration=0.5, sample_rate=20000, is_faulty=False, random_seed=42)
        faulty = generate_vibration_signal(duration=0.5, sample_rate=20000, is_faulty=True, random_seed=42)

        healthy_peak = np.max(np.abs(healthy[:, 0]))
        faulty_peak = np.max(np.abs(faulty[:, 0]))
        self.assertGreater(faulty_peak, healthy_peak)

    def test_save_and_load_nasa_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_sample.txt"
            data = generate_vibration_signal(duration=0.1, sample_rate=20000, is_faulty=False, random_seed=12)
            save_nasa_format_file(data, tmp_path)

            self.assertTrue(tmp_path.exists())
            loaded = np.loadtxt(tmp_path, delimiter="\t")
            self.assertEqual(loaded.shape, (2000, 4))


if __name__ == "__main__":
    unittest.main()
