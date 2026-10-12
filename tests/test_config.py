"""Unit tests for EchoGuard configuration and path resolution."""

import unittest
from pathlib import Path
from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SAMPLES_DATA_PATH,
    SAMPLE_RATE,
    IMG_HEIGHT,
    IMG_WIDTH
)


class TestConfig(unittest.TestCase):
    def test_project_root_exists(self):
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue((PROJECT_ROOT / "src").is_dir())

    def test_data_paths_defined(self):
        self.assertIsInstance(DATA_DIR, Path)
        self.assertIsInstance(RAW_DATA_PATH, Path)
        self.assertIsInstance(PROCESSED_DATA_PATH, Path)
        self.assertIsInstance(SAMPLES_DATA_PATH, Path)

    def test_default_hyperparameters(self):
        self.assertEqual(SAMPLE_RATE, 20000)
        self.assertEqual(IMG_HEIGHT, 256)
        self.assertEqual(IMG_WIDTH, 256)


if __name__ == "__main__":
    unittest.main()
