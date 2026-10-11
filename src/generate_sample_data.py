"""
EchoGuard Synthetic Vibration Signal Generator
Generates realistic 4-channel vibration data mimicking the NASA Bearing Dataset
at 20 kHz sampling frequency for demonstration, offline testing, and CI pipelines.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import SAMPLE_RATE, SAMPLES_DATA_PATH, RAW_DATA_PATH


def generate_vibration_signal(
    duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    is_faulty: bool = False,
    noise_level: float = 0.05,
    random_seed: int = None
) -> np.ndarray:
    """
    Generates synthetic 4-channel vibration acceleration signal in NASA format.

    Channel 0: Target bearing under observation.
    Channels 1-3: Adjacent auxiliary bearing sensor channels.

    Args:
        duration: Length of recording in seconds.
        sample_rate: Sampling frequency in Hz (NASA sensor standard = 20,000 Hz).
        is_faulty: If True, introduces defect impulse impacts and elevated harmonics.
        noise_level: Amplitude of baseline ambient Gaussian noise.
        random_seed: Optional seed for reproducible generation.

    Returns:
        np.ndarray of shape (num_samples, 4).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Fundamental shaft rotation frequency ~33.3 Hz (approx 2000 RPM)
    f_shaft = 33.33
    shaft_harmonic = 0.08 * np.sin(2 * np.pi * f_shaft * t) + 0.03 * np.sin(2 * np.pi * 2 * f_shaft * t)

    if not is_faulty:
        # Healthy: Low amplitude vibration dominated by rotational frequency and white noise
        b1 = shaft_harmonic + np.random.normal(0, noise_level, num_samples)
        b2 = 0.05 * np.sin(2 * np.pi * f_shaft * t) + np.random.normal(0, noise_level, num_samples)
        b3 = 0.04 * np.sin(2 * np.pi * f_shaft * t) + np.random.normal(0, noise_level, num_samples)
        b4 = 0.05 * np.sin(2 * np.pi * f_shaft * t) + np.random.normal(0, noise_level, num_samples)
    else:
        # Faulty: Ball pass frequency outer race (BPFO ~ 236 Hz) periodic transient impacts
        f_bpfo = 236.4
        impact_period = int(sample_rate / f_bpfo)
        impulses = np.zeros(num_samples)

        # Generate decaying exponential shockwaves at defect frequency
        decay_window = int(0.003 * sample_rate)  # 3ms impulse ring-down
        decay_profile = np.exp(-np.linspace(0, 5, decay_window)) * np.sin(2 * np.pi * 3200 * np.linspace(0, 0.003, decay_window))

        for idx in range(0, num_samples - decay_window, impact_period):
            jitter = np.random.randint(-5, 6)
            pos = max(0, min(num_samples - decay_window, idx + jitter))
            impulses[pos:pos + decay_window] += (0.8 + np.random.uniform(-0.15, 0.15)) * decay_profile

        # High variance + shock impacts + harmonic distortion
        b1 = shaft_harmonic + impulses + np.random.normal(0, noise_level * 2.5, num_samples)
        b2 = 0.25 * impulses + np.random.normal(0, noise_level * 1.5, num_samples)
        b3 = 0.15 * impulses + np.random.normal(0, noise_level * 1.2, num_samples)
        b4 = 0.10 * impulses + np.random.normal(0, noise_level, num_samples)

    # Stack into 4-column matrix matching NASA format
    channels = np.column_stack([b1, b2, b3, b4])
    return channels


def save_nasa_format_file(data: np.ndarray, file_path: Path):
    """
    Saves multi-channel vibration array in tab-separated format without header.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, data, delimiter="\t", fmt="%.6f")


def generate_samples():
    """Generates standalone healthy and faulty demo samples for UI and testing."""
    SAMPLES_DATA_PATH.mkdir(parents=True, exist_ok=True)

    healthy_file = SAMPLES_DATA_PATH / "healthy_sample.txt"
    faulty_file = SAMPLES_DATA_PATH / "faulty_sample.txt"

    print("Generating healthy demo sample...")
    healthy_data = generate_vibration_signal(duration=1.0, is_faulty=False, random_seed=42)
    save_nasa_format_file(healthy_data, healthy_file)
    print(f"Saved: {healthy_file} ({healthy_data.shape[0]} rows)")

    print("Generating faulty demo sample...")
    faulty_data = generate_vibration_signal(duration=1.0, is_faulty=True, random_seed=101)
    save_nasa_format_file(faulty_data, faulty_file)
    print(f"Saved: {faulty_file} ({faulty_data.shape[0]} rows)")


def generate_raw_dataset(count: int = 10):
    """Generates a small batch of raw dataset files in data/raw for local pipeline testing."""
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Generating {count} raw test files in {RAW_DATA_PATH}...")

    # First half healthy, second half faulty
    half = count // 2
    for i in range(half):
        data = generate_vibration_signal(duration=0.5, is_faulty=False, random_seed=1000 + i)
        save_nasa_format_file(data, RAW_DATA_PATH / f"2004.02.12.00.{i:02d}.00")

    for i in range(half):
        data = generate_vibration_signal(duration=0.5, is_faulty=True, random_seed=2000 + i)
        save_nasa_format_file(data, RAW_DATA_PATH / f"2004.02.19.00.{i:02d}.00")

    print(f"Generated {count} raw dataset files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGuard Synthetic Vibration Signal Generator")
    parser.add_argument("--mode", choices=["samples", "raw", "all"], default="samples",
                        help="Generate demo samples or raw dataset files")
    parser.add_argument("--count", type=int, default=10, help="Number of raw files to generate if mode is 'raw' or 'all'")
    args = parser.parse_args()

    if args.mode in ("samples", "all"):
        generate_samples()
    if args.mode in ("raw", "all"):
        generate_raw_dataset(args.count)
