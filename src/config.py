"""
EchoGuard Configuration Module
Provides centralized paths, signal processing parameters, and model hyperparameters
with automatic project root resolution and environment overrides.
"""

import os
from pathlib import Path

# --- DIRECTORY PATHS ---
# Resolves project root regardless of the current working directory
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = Path(os.getenv("ECHOGUARD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
SAMPLES_DATA_PATH = DATA_DIR / "samples"

HEALTHY_DIR = PROCESSED_DATA_PATH / "healthy"
FAULTY_DIR = PROCESSED_DATA_PATH / "faulty"

MODELS_DIR = Path(os.getenv("ECHOGUARD_MODELS_DIR", PROJECT_ROOT / "models"))
DEFAULT_MODEL_PATH = MODELS_DIR / "echo_guard_model.keras"

# Ensure essential output directories exist
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
FAULTY_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DATA_PATH.mkdir(parents=True, exist_ok=True)

# --- SIGNAL PROCESSING SETTINGS ---
# NASA bearing test sensor sample rate is 20,000 Hz (20 kHz)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "20000"))
N_FFT = int(os.getenv("N_FFT", "2048"))
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "512"))

# --- MODEL TRAINING HYPERPARAMETERS ---
IMG_HEIGHT = int(os.getenv("IMG_HEIGHT", "256"))
IMG_WIDTH = int(os.getenv("IMG_WIDTH", "256"))
IMG_CHANNELS = 3
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "15"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.5"))
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 123
