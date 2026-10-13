"""
EchoGuard Vibration Feature Extraction & ISO Condition Monitoring Engine
Computes time-domain statistical indicators, frequency-domain FFT harmonics,
and ISO 10816 machinery vibration severity classification.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np


def extract_time_domain_features(signal: np.ndarray, eps: float = 1e-9) -> Dict[str, float]:
    """
    Computes time-domain statistical vibration indicators from a 1D signal.

    Args:
        signal: 1D array of acceleration/vibration measurements.
        eps: Small constant to avoid division by zero.

    Returns:
        Dict containing statistical indicators:
          - peak: Maximum absolute amplitude
          - p2p: Peak-to-peak amplitude (max - min)
          - rms: Root Mean Square acceleration
          - mean: Arithmetic mean
          - variance: Variance
          - std: Standard deviation
          - crest_factor: Peak / RMS (transient impulsiveness)
          - kurtosis: 4th normalized central moment (spalling / surface fatigue indicator)
          - skewness: 3rd normalized central moment (asymmetry)
          - shape_factor: RMS / Mean Absolute Deviation
          - impulse_factor: Peak / Mean Absolute Deviation
          - clearance_factor: Peak / (Mean(sqrt(|x|)))^2
    """
    signal = np.asarray(signal, dtype=float).flatten()
    if len(signal) == 0:
        raise ValueError("Signal array cannot be empty.")

    abs_sig = np.abs(signal)
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    var_val = float(np.var(signal))
    peak_val = float(np.max(abs_sig))
    p2p_val = float(np.ptp(signal))
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    mean_abs = float(np.mean(abs_sig))
    square_root_abs = float(np.mean(np.sqrt(abs_sig)))

    # Central moments
    diff = signal - mean_val
    m3 = float(np.mean(diff ** 3))
    m4 = float(np.mean(diff ** 4))

    skewness = float(m3 / (std_val ** 3 + eps)) if std_val > 0 else 0.0
    kurtosis = float(m4 / (std_val ** 4 + eps)) if std_val > 0 else 0.0

    crest_factor = float(peak_val / (rms_val + eps))
    shape_factor = float(rms_val / (mean_abs + eps))
    impulse_factor = float(peak_val / (mean_abs + eps))
    clearance_factor = float(peak_val / ((square_root_abs ** 2) + eps))

    return {
        "peak": round(peak_val, 4),
        "p2p": round(p2p_val, 4),
        "rms": round(rms_val, 4),
        "mean": round(mean_val, 4),
        "variance": round(var_val, 4),
        "std": round(std_val, 4),
        "crest_factor": round(crest_factor, 4),
        "kurtosis": round(kurtosis, 4),
        "skewness": round(skewness, 4),
        "shape_factor": round(shape_factor, 4),
        "impulse_factor": round(impulse_factor, 4),
        "clearance_factor": round(clearance_factor, 4),
    }


def compute_fft_spectrum(
    signal: np.ndarray,
    sample_rate: int = 20000,
    top_k_peaks: int = 5
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes single-sided Fast Fourier Transform (FFT) amplitude spectrum.

    Args:
        signal: 1D vibration array.
        sample_rate: Sampling frequency in Hz.
        top_k_peaks: Number of dominant harmonic peaks to identify.

    Returns:
        Tuple of (frequencies, amplitudes, spectral_metrics_dict)
    """
    signal = np.asarray(signal, dtype=float).flatten()
    n = len(signal)
    if n == 0:
        raise ValueError("Signal array cannot be empty.")

    # Remove DC bias component
    detrended = signal - np.mean(signal)

    # Compute FFT
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    amps = np.abs(fft_vals) * 2.0 / n

    total_energy = float(np.sum(amps ** 2))
    sum_amps = float(np.sum(amps)) + 1e-9

    # Spectral Centroid: center of gravity of spectrum
    spectral_centroid = float(np.sum(freqs * amps) / sum_amps)

    # Dominant peak frequency
    dom_idx = int(np.argmax(amps))
    dominant_frequency = float(freqs[dom_idx])
    dominant_amplitude = float(amps[dom_idx])

    # Top-K peaks
    sorted_indices = np.argsort(amps)[::-1][:top_k_peaks]
    dominant_peaks = [
        {"frequency_hz": round(float(freqs[idx]), 2), "amplitude": round(float(amps[idx]), 4)}
        for idx in sorted_indices
    ]

    # Frequency band energy distribution
    band_sub = float(np.sum(amps[freqs < 20] ** 2))
    band_shaft = float(np.sum(amps[(freqs >= 20) & (freqs < 100)] ** 2))
    band_defect = float(np.sum(amps[(freqs >= 100) & (freqs < 1000)] ** 2))
    band_hf = float(np.sum(amps[freqs >= 1000] ** 2))

    norm_factor = total_energy + 1e-9
    spectral_metrics = {
        "dominant_frequency_hz": round(dominant_frequency, 2),
        "dominant_amplitude": round(dominant_amplitude, 4),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "total_spectral_energy": round(total_energy, 4),
        "dominant_peaks": dominant_peaks,
        "energy_bands": {
            "sub_harmonic_ratio": round(band_sub / norm_factor, 4),
            "rotational_shaft_ratio": round(band_shaft / norm_factor, 4),
            "defect_band_ratio": round(band_defect / norm_factor, 4),
            "high_frequency_impact_ratio": round(band_hf / norm_factor, 4),
        }
    }

    return freqs, amps, spectral_metrics


# ISO 10816-3 Vibration Severity Thresholds (RMS velocity in mm/s or g-equivalent standard)
# Classes:
#   Class I: Small electrical motors / pumps (< 15 kW)
#   Class II: Medium machines (15 kW - 75 kW)
#   Class III: Large machines with rigid foundation (> 75 kW)
#   Class IV: Large machines with soft / flexible foundation
ISO_10816_THRESHOLDS = {
    "Class I": {"good": 0.71, "acceptable": 1.80, "alert": 4.50},
    "Class II": {"good": 1.12, "acceptable": 2.80, "alert": 7.10},
    "Class III": {"good": 1.80, "acceptable": 4.50, "alert": 11.20},
    "Class IV": {"good": 2.80, "acceptable": 7.10, "alert": 18.00},
}


def classify_iso_severity(
    rms_value: float,
    machine_class: str = "Class II"
) -> Dict[str, Any]:
    """
    Evaluates vibration severity based on the ISO 10816-3 machinery standard.

    Zones:
      - Zone A: Newly commissioned machines / Good condition.
      - Zone B: Continuous unrestricted operation permissible.
      - Zone C: Unsatisfactory; restricted continuous operation; alert condition.
      - Zone D: Unacceptable; vibration causes damage; immediate shutdown required.

    Args:
        rms_value: Overall vibration RMS magnitude.
        machine_class: Machine classification ("Class I", "Class II", "Class III", "Class IV").

    Returns:
        Dict with zone, status, description, color code, and threshold bounds.
    """
    thresholds = ISO_10816_THRESHOLDS.get(machine_class, ISO_10816_THRESHOLDS["Class II"])

    if rms_value <= thresholds["good"]:
        zone = "Zone A"
        status = "Good"
        description = "Vibration is typical of new or freshly overhauled machines."
        action = "Standard routine operational monitoring."
        color = "#16A34A"  # Green
        severity_score = 1
    elif rms_value <= thresholds["acceptable"]:
        zone = "Zone B"
        status = "Acceptable"
        description = "Vibration acceptable for long-term continuous operation."
        action = "Continue standard operating schedule."
        color = "#65A30D"  # Lime Green
        severity_score = 2
    elif rms_value <= thresholds["alert"]:
        zone = "Zone C"
        status = "Unsatisfactory / Alert"
        description = "Vibration is elevated; sustained operation may cause fatigue damage."
        action = "Schedule maintenance inspection; investigate defect frequencies."
        color = "#D97706"  # Amber
        severity_score = 3
    else:
        zone = "Zone D"
        status = "Unacceptable / Critical"
        description = "Dangerous vibration level; high probability of catastrophic failure."
        action = "Immediate planned shutdown and assembly overhaul recommended."
        color = "#DC2626"  # Red
        severity_score = 4

    return {
        "machine_class": machine_class,
        "rms_value": round(rms_value, 4),
        "zone": zone,
        "status": status,
        "description": description,
        "recommended_action": action,
        "color": color,
        "severity_score": severity_score,
        "thresholds": thresholds
    }


def extract_all_vibration_features(
    signal: np.ndarray,
    sample_rate: int = 20000,
    machine_class: str = "Class II"
) -> Dict[str, Any]:
    """
    Extracts the complete feature profile: time-domain indicators, FFT spectrum metrics,
    and ISO 10816 severity classification.

    Args:
        signal: 1D vibration signal array.
        sample_rate: Sampling frequency in Hz.
        machine_class: Machine classification standard.

    Returns:
        Nested dictionary containing comprehensive diagnostic features.
    """
    time_features = extract_time_domain_features(signal)
    _, _, spectral_metrics = compute_fft_spectrum(signal, sample_rate=sample_rate)
    iso_result = classify_iso_severity(time_features["rms"], machine_class=machine_class)

    return {
        "time_domain": time_features,
        "frequency_domain": spectral_metrics,
        "iso_severity": iso_result,
        "sample_count": len(signal),
        "sample_rate": sample_rate,
        "duration_seconds": round(len(signal) / sample_rate, 4),
    }
