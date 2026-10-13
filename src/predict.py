"""
EchoGuard CLI Batch Inference & Diagnostic Reporting Tool
Evaluates single vibration sensor files or entire telemetry directories,
computing AI CNN fault diagnosis, time/frequency domain KPIs, and ISO 10816 status.
"""

import os
import sys
import argparse
import json
import csv
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEFAULT_MODEL_PATH,
    SAMPLE_RATE,
    IMG_HEIGHT,
    IMG_WIDTH,
    N_FFT,
    HOP_LENGTH,
)
from src.features import extract_all_vibration_features

try:
    import tensorflow as tf
    from models.model import load_echo_guard_model, build_echo_guard_cnn
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def load_vibration_file(file_path: Path, channel: int = 0, sample_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Loads 1D vibration array from .txt, .csv, or .wav file.

    Args:
        file_path: Path to target file.
        channel: Column index to extract (default 0).
        sample_rate: Sampling frequency in Hz.

    Returns:
        Tuple of (1D numpy array, effective sample rate).
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path_obj.suffix.lower()

    if suffix in (".wav", ".mp3", ".flac"):
        try:
            import librosa
            sig, sr = librosa.load(str(path_obj), sr=sample_rate)
            return sig, sr
        except ImportError:
            import wave
            with wave.open(str(path_obj), 'rb') as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                sig = np.frombuffer(frames, dtype=np.int16).astype(float) / 32768.0
            return sig, sr

    # NASA tab/space delimited format or CSV
    sep = r'\s+' if suffix != ".csv" else ','
    try:
        df = pd.read_csv(str(path_obj), sep=sep, header=None, engine='python')
    except Exception:
        # Fallback to pure numpy loading
        data = np.loadtxt(str(path_obj))
        if data.ndim == 1:
            return data.astype(float), sample_rate
        return data[:, min(channel, data.shape[1] - 1)].astype(float), sample_rate

    target_col = min(channel, len(df.columns) - 1)
    sig = df[target_col].values.astype(float)
    return sig, sample_rate


def predict_from_signal(
    signal: np.ndarray,
    model: Optional[Any] = None,
    sample_rate: int = SAMPLE_RATE,
    machine_class: str = "Class II"
) -> Dict[str, Any]:
    """
    Performs complete predictive maintenance analysis on a 1D vibration signal.

    Args:
        signal: 1D vibration acceleration data.
        model: Loaded tf.keras.Model or None.
        sample_rate: Sampling rate in Hz.
        machine_class: ISO 10816 machine class.

    Returns:
        Structured diagnostic dictionary.
    """
    # 1. Extract physical features & ISO classification
    features = extract_all_vibration_features(signal, sample_rate=sample_rate, machine_class=machine_class)
    rms = features["time_domain"]["rms"]
    kurtosis = features["time_domain"]["kurtosis"]
    crest = features["time_domain"]["crest_factor"]

    # 2. Neural Network Inference
    confidence: float
    if TF_AVAILABLE and model is not None:
        try:
            import librosa
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Compute STFT Mel-spectrogram
            D = librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(1, 1, 1)
            librosa.display.specshow(S_db, sr=sample_rate, ax=ax)
            plt.axis("off")
            plt.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            img = tf.keras.utils.load_img(tmp_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img)
            img_batch = tf.expand_dims(img_array, 0)
            pred = model.predict(img_batch, verbose=0)
            confidence = float(pred[0][0])

            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            # Fallback to feature-derived health indicator
            confidence = _heuristic_confidence(kurtosis, crest, rms)
    else:
        # Physics-based baseline indicator when TensorFlow model is not loaded
        confidence = _heuristic_confidence(kurtosis, crest, rms)

    # In EchoGuard: Class 0 = Faulty (< 0.5), Class 1 = Healthy (>= 0.5)
    is_healthy = bool(confidence >= 0.5)
    status_label = "HEALTHY" if is_healthy else "FAULTY"
    severity_state = "NORMAL" if is_healthy else "CRITICAL_ALERT"

    return {
        "prediction": status_label,
        "is_healthy": is_healthy,
        "confidence": round(confidence, 4),
        "health_score_pct": int(confidence * 100),
        "system_status": severity_state,
        "iso_severity": features["iso_severity"],
        "key_metrics": {
            "rms": rms,
            "peak": features["time_domain"]["peak"],
            "kurtosis": kurtosis,
            "crest_factor": crest,
            "dominant_frequency_hz": features["frequency_domain"]["dominant_frequency_hz"],
        },
        "full_features": features,
    }


def _heuristic_confidence(kurtosis: float, crest: float, rms: float) -> float:
    """
    Robust physics-based vibration anomaly index based on Kurtosis and Crest factor.
    Normal gaussian vibration: Kurtosis ~ 3.0, Crest ~ 3.0.
    Impulsive bearing defect: Kurtosis > 3.5, Crest > 4.0.
    """
    anomaly_penalty = 0.0
    if kurtosis > 3.3:
        anomaly_penalty += (kurtosis - 3.0) * 0.25
    if crest > 3.8:
        anomaly_penalty += (crest - 3.8) * 0.20
    if rms > 0.18:
        anomaly_penalty += (rms - 0.18) * 0.8

    health_prob = max(0.02, min(0.98, 0.95 - anomaly_penalty))
    return float(health_prob)


def batch_predict(
    file_paths: List[Path],
    model: Optional[Any] = None,
    channel: int = 0,
    sample_rate: int = SAMPLE_RATE,
    machine_class: str = "Class II"
) -> List[Dict[str, Any]]:
    """
    Executes diagnosis on a sequence of files and aggregates results.
    """
    results = []
    for fp in file_paths:
        try:
            sig, sr = load_vibration_file(fp, channel=channel, sample_rate=sample_rate)
            diag = predict_from_signal(sig, model=model, sample_rate=sr, machine_class=machine_class)
            diag["file_path"] = str(fp)
            diag["filename"] = fp.name
            results.append(diag)
        except Exception as exc:
            results.append({
                "file_path": str(fp),
                "filename": fp.name,
                "error": str(exc),
                "prediction": "ERROR",
                "is_healthy": False,
                "confidence": 0.0,
                "health_score_pct": 0,
                "system_status": "FAILED",
                "iso_severity": {"zone": "N/A", "status": "Error"},
                "key_metrics": {}
            })
    return results


def format_cli_table(results: List[Dict[str, Any]]) -> str:
    """Formats batch results into an aligned terminal summary table."""
    header = f"{'FILENAME':<28} | {'PREDICTION':<10} | {'CONFIDENCE':<10} | {'RMS (g)':<9} | {'KURTOSIS':<9} | {'ISO ZONE':<8}"
    divider = "-" * len(header)
    lines = [divider, header, divider]

    for r in results:
        fname = r.get("filename", "unknown")
        if len(fname) > 26:
            fname = fname[:23] + "..."
        pred = r.get("prediction", "N/A")
        conf = f"{r.get('confidence', 0.0) * 100:.1f}%"
        metrics = r.get("key_metrics", {})
        rms_str = f"{metrics.get('rms', 0.0):.4f}" if "rms" in metrics else "N/A"
        kurt_str = f"{metrics.get('kurtosis', 0.0):.2f}" if "kurtosis" in metrics else "N/A"
        zone = r.get("iso_severity", {}).get("zone", "N/A")
        lines.append(f"{fname:<28} | {pred:<10} | {conf:<10} | {rms_str:<9} | {kurt_str:<9} | {zone:<8}")

    lines.append(divider)
    return "\n".join(lines)


def run_cli():
    parser = argparse.ArgumentParser(description="EchoGuard CLI Batch Inference & Diagnostic Reporter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str, help="Path to a single vibration file (.txt, .csv, .wav)")
    group.add_argument("--input-dir", "-d", type=str, help="Directory containing vibration files for batch diagnosis")

    parser.add_argument("--model-path", "-m", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to pre-trained .keras model")
    parser.add_argument("--channel", "-c", type=int, default=0, help="Accelerometer channel index (default 0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=SAMPLE_RATE, help="Sampling frequency in Hz (default 20000)")
    parser.add_argument("--machine-class", type=str, default="Class II", choices=["Class I", "Class II", "Class III", "Class IV"])
    parser.add_argument("--output-json", type=str, help="Filepath to export results as structured JSON")
    parser.add_argument("--output-csv", type=str, help="Filepath to export summary table as CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose stdout table output")

    args = parser.parse_args()

    # Discover files
    files_to_process: List[Path] = []
    if args.input:
        p = Path(args.input)
        if not p.is_file():
            print(f"Error: Specified file does not exist: {p}", file=sys.stderr)
            sys.exit(1)
        files_to_process.append(p)
    elif args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.is_dir():
            print(f"Error: Specified directory does not exist: {dir_path}", file=sys.stderr)
            sys.exit(1)
        valid_exts = {".txt", ".csv", ".wav", ".mp3", ""}
        files_to_process = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() in valid_exts or not f.suffix)
        ])
        if not files_to_process:
            print(f"No suitable vibration files found in {dir_path}", file=sys.stderr)
            sys.exit(1)

    # Load Model
    model = None
    if TF_AVAILABLE and Path(args.model_path).exists():
        model = load_echo_guard_model(args.model_path)

    # Execute Batch Prediction
    results = batch_predict(
        files_to_process,
        model=model,
        channel=args.channel,
        sample_rate=args.sample_rate,
        machine_class=args.machine_class
    )

    # Terminal output
    if not args.quiet:
        print("\n" + "=" * 65)
        print("🔊 EchoGuard Industrial Predictive Maintenance Report")
        print("=" * 65)
        print(format_cli_table(results))
        fault_count = sum(1 for r in results if r.get("prediction") == "FAULTY")
        healthy_count = sum(1 for r in results if r.get("prediction") == "HEALTHY")
        print(f"\nSummary: {len(results)} files analyzed | {healthy_count} Healthy | {fault_count} Faulty")

    # Export JSON
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Exported JSON diagnostic report to: {out_p}")

    # Export CSV
    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Prediction", "Confidence", "HealthScorePct", "Status", "ISO_Zone", "RMS", "Kurtosis", "CrestFactor", "DominantFreqHz"])
            for r in results:
                km = r.get("key_metrics", {})
                writer.writerow([
                    r.get("filename", ""),
                    r.get("prediction", ""),
                    r.get("confidence", 0.0),
                    r.get("health_score_pct", 0),
                    r.get("system_status", ""),
                    r.get("iso_severity", {}).get("zone", ""),
                    km.get("rms", ""),
                    km.get("kurtosis", ""),
                    km.get("crest_factor", ""),
                    km.get("dominant_frequency_hz", "")
                ])
        if not args.quiet:
            print(f"Exported CSV summary to: {out_p}")

    # Exit code: 2 if any faulty bearing detected, 0 if all healthy
    any_fault = any(r.get("prediction") == "FAULTY" for r in results)
    sys.exit(2 if any_fault else 0)


if __name__ == "__main__":
    run_cli()
