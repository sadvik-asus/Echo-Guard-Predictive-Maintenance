"""Unit tests for EchoGuard CNN model architecture."""

import unittest

try:
    import tensorflow as tf
    from models.model import build_echo_guard_cnn, load_echo_guard_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestEchoGuardModel(unittest.TestCase):
    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed in current environment")
    def test_build_echo_guard_cnn(self):
        model = build_echo_guard_cnn(input_shape=(256, 256, 3), num_classes=1, compile_model=True)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "EchoGuard_CNN")
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(model.input_shape, (None, 256, 256, 3))

    @unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed in current environment")
    def test_load_nonexistent_model(self):
        res = load_echo_guard_model("nonexistent_model_bundle.keras")
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()
