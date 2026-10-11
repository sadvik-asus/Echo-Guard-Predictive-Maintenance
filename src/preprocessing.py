import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    HEALTHY_DIR,
    FAULTY_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
)

def generate_spectrogram_image(file_path, save_path, sample_rate=SAMPLE_RATE):
    """
    Reads a raw vibration file, computes the spectrogram, and saves it as an image.

    Args:
        file_path: Path to raw vibration data file (.txt, .csv) or audio (.wav).
        save_path: Destination path for spectrogram PNG image.
        sample_rate: Sampling frequency in Hz (default 20,000 Hz).

    Returns:
        bool: True if spectrogram was generated and saved successfully, False otherwise.
    """
    try:
        file_str = str(file_path).lower()
        if file_str.endswith(('.wav', '.mp3', '.flac')):
            signal, sr = librosa.load(file_path, sr=sample_rate)
        else:
            # 1. Read the data (NASA format is tab-separated, no header)
            df = pd.read_csv(file_path, sep='\t', header=None)
            # 2. Extract Bearing 1 data (Column 0)
            signal = df[0].values
            sr = sample_rate

        # 3. Short-Time Fourier Transform (STFT)
        D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # 4. Save as Image (No axes, just the data pattern)
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_db, sr=sr)
        plt.axis('off')

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def process_dataset():
    """
    Processes raw bearing vibration files into healthy and faulty spectrogram sets.
    """
    if not RAW_DATA_PATH.exists():
        print(f"Directory {RAW_DATA_PATH} does not exist.")
        return

    # Filter out hidden files like .gitkeep
    files = sorted([f for f in os.listdir(RAW_DATA_PATH) if not f.startswith('.')])
    if not files:
        print(f"No raw data files found in {RAW_DATA_PATH}.")
        print("Tip: Run 'python src/generate_sample_data.py' to generate synthetic vibration files.")
        return

    # NASA 2nd_test strategy: early files are healthy, tail files are faulty
    split_count = min(300, max(1, len(files) // 2))
    healthy_files = files[:split_count]
    faulty_files = files[-split_count:] if len(files) > 1 else []

    print(f"Processing {len(healthy_files)} Healthy files...")
    for i, f in enumerate(healthy_files):
        source = RAW_DATA_PATH / f
        target = HEALTHY_DIR / f"{i}.png"
        generate_spectrogram_image(source, target)
        if i % 50 == 0:
            print(f"  Processed {i} healthy...")

    print(f"Processing {len(faulty_files)} Faulty files...")
    for i, f in enumerate(faulty_files):
        source = RAW_DATA_PATH / f
        target = FAULTY_DIR / f"{i}.png"
        generate_spectrogram_image(source, target)
        if i % 10 == 0:
            print(f"  Processed {i} faulty...")

    print(f"Done! Check your {PROCESSED_DATA_PATH} folder.")

if __name__ == "__main__":
    process_dataset()