"""Voice activity detection from short-time energy and spectral shape."""

from __future__ import annotations

import numpy as np


def detect_speech(features: dict, method: str = "energy_spectral") -> tuple[np.ndarray, dict]:
    """Return a boolean speech mask aligned with analysis frames."""
    ste = features["ste"]
    flatness = features["flatness"]
    centroid = features["centroid"]
    if len(ste) == 0:
        raise ValueError("No frames were available for voice activity detection.")

    low = float(np.percentile(ste, 20))
    high = float(np.percentile(ste, 85))
    threshold = max(low * 3.5, high * 0.045, 1e-6)
    energetic = ste > threshold
    strong = ste > threshold * 2.8

    if method == "energy":
        mask = energetic
        method_name = "Energy"
    else:
        spectral_ok = (flatness < 0.55) & (centroid > 180.0) & (centroid < 5500.0)
        mask = strong | (energetic & spectral_ok)
        method_name = "Energy + Spectral"

    mask = _median_bool(mask, 5)
    speech_ratio = float(mask.mean()) if len(mask) else 0.0
    return mask, {
        "method": method_name,
        "threshold": threshold,
        "speech_ratio": speech_ratio,
        "frames": int(len(mask)),
    }


def _median_bool(mask: np.ndarray, width: int) -> np.ndarray:
    if width < 3 or len(mask) < width:
        return mask.copy()
    kernel = np.ones(width, dtype=np.float32) / width
    smoothed = np.convolve(mask.astype(np.float32), kernel, mode="same")
    return smoothed >= 0.5
