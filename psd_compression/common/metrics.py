from __future__ import annotations

import numpy as np


def mse(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(diff * diff))


def nmse(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    den = float(np.mean(x * x) + eps)
    return float(np.mean((x - y) ** 2) / den)


def snr_db(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    signal = float(np.mean(x * x) + eps)
    noise = float(np.mean((x - y) ** 2) + eps)
    return float(10.0 * np.log10(signal / noise))


def occupancy_mask(psd: np.ndarray, margin_db: float = 3.0, eps: float = 1e-30) -> np.ndarray:
    """Simple occupancy mask above median + margin in dB."""
    psd = np.asarray(psd, dtype=np.float64)
    psd_db = 10.0 * np.log10(np.maximum(psd, eps))
    threshold = float(np.median(psd_db) + margin_db)
    return psd_db > threshold


def occupancy_mismatch_rate(original: np.ndarray, reconstructed: np.ndarray, margin_db: float = 3.0) -> float:
    o = occupancy_mask(original, margin_db=margin_db)
    r = occupancy_mask(reconstructed, margin_db=margin_db)
    return float(np.mean(o != r))

