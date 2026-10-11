"""
EchoGuard CNN Architecture Definition
Provides modular model construction, compilation, and utility functions
for frequency-domain spectrogram classification in industrial predictive maintenance.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics


def build_echo_guard_cnn(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
    compile_model: bool = True
) -> tf.keras.Model:
    """
    Constructs the custom 2D Convolutional Neural Network for vibration Mel-spectrograms.

    Architecture:
      - Rescaling: Normalizes pixel values from [0, 255] to [0, 1].
      - Conv Block 1: 16 filters, 3x3 kernel, ReLU, MaxPooling2D (2x2).
      - Conv Block 2: 32 filters, 3x3 kernel, ReLU, MaxPooling2D (2x2).
      - Conv Block 3: 64 filters, 3x3 kernel, ReLU, MaxPooling2D (2x2).
      - Dense Classifier Head: Flatten -> Dense(128, ReLU) -> Dropout(rate) -> Dense(num_classes).

    Args:
        input_shape: Dimensions of spectrogram image tensor (height, width, channels).
        num_classes: Number of target output classes (default 1 for binary fault classification).
        dropout_rate: Dropout probability to mitigate overfitting on small acoustic samples.
        learning_rate: Learning rate for Adam optimizer.
        compile_model: Whether to compile model with loss and optimizer.

    Returns:
        Compiled or uncompiled tf.keras.Model instance.
    """
    output_activation = "sigmoid" if num_classes == 1 else "softmax"
    loss_fn = losses.BinaryCrossentropy() if num_classes == 1 else losses.CategoricalCrossentropy()

    model = models.Sequential([
        # 0. Input Preprocessing
        layers.Rescaling(1.0 / 255, input_shape=input_shape, name="pixel_rescaling"),

        # 1. Feature Extraction Block 1
        layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        # 2. Feature Extraction Block 2
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        # 3. Feature Extraction Block 3
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        # 4. Dense Classification Head
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_features"),
        layers.Dropout(dropout_rate, name="dropout_regularization"),
        layers.Dense(num_classes, activation=output_activation, name="classification_output")
    ], name="EchoGuard_CNN")

    if compile_model:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall")]
        )

    return model


def load_echo_guard_model(model_path: str) -> Optional[tf.keras.Model]:
    """
    Safely loads a pre-trained Keras model from disk.

    Args:
        model_path: Filepath to the .keras or SavedModel bundle.

    Returns:
        Loaded tf.keras.Model or None if file cannot be loaded.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as exc:
        print(f"[EchoGuard] Warning: Failed to load model from {model_path}: {exc}")
        return None
