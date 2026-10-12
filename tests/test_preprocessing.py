"""Unit tests for vibration preprocessing and Mel-spectrogram generation."""

import unittest
import tempfile
from pathlib import Path
from src.generate_sample_data import generate_vibration_signal, save_nasa_format_file

try:
    import librosa
    import matplotlib.pyplot as plt
    from src.preprocessing import generate_spectrogram_image
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class TestPreprocessing(unittest.TestCase):
    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa/matplotlib not installed in current environment")
    def test_generate_spectrogram_from_nasa_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_file = Path(tmpdir) / "sample.txt"
            spec_file = Path(tmpdir) / "spec.png"

            data = generate_vibration_signal(duration=0.1, sample_rate=20000, is_faulty=False, random_seed=99)
            save_nasa_format_file(data, raw_file)

            success = generate_spectrogram_image(raw_file, spec_file)
            self.assertTrue(success)
            self.assertTrue(spec_file.exists())
            self.assertGreater(spec_file.stat().st_size, 0)

    @unittest.skipUnless(LIBROSA_AVAILABLE, "librosa/matplotlib not installed in current environment")
    def test_generate_spectrogram_invalid_file_handling(self):
        non_existent = Path("non_existent_file_path_12345.txt")
        output = Path("should_not_exist.png")
        success = generate_spectrogram_image(non_existent, output)
        self.assertFalse(success)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
