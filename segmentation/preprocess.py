"""Resample, denoise, and normalize speech before framing."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from segmentation.config import ANALYSIS_SR


def prepare_audio(audio: np.ndarray, sample_rate: int, denoise: bool = True) -> tuple[np.ndarray, list[str]]:
    """Return 16 kHz mono float audio and a short list of processing notes."""
    notes: list[str] = []
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(y) < int(sample_rate * 0.2):
        raise ValueError("Audio is too short. Record or upload at least 0.2 seconds.")

    y = y - float(np.mean(y))
    y = _resample(y, sample_rate, ANALYSIS_SR)
    if sample_rate != ANALYSIS_SR:
        notes.append(f"Resampled from {sample_rate} Hz to {ANALYSIS_SR} Hz for analysis.")
    else:
        notes.append(f"Analysis rate is already {ANALYSIS_SR} Hz.")

    if len(y) > 400:
        sos = signal.butter(4, 80.0, btype="highpass", fs=ANALYSIS_SR, output="sos")
        y = signal.sosfiltfilt(sos, y).astype(np.float32)
        notes.append("High-pass filtered at 80 Hz.")

    if denoise:
        y = _spectral_gate(y)
        notes.append("Applied a mild spectral noise gate.")
    else:
        notes.append("Spectral noise gate skipped.")

    y = _rms_normalize(y)
    y = _fade(y, ANALYSIS_SR)
    notes.append("Removed DC offset and normalized loudness.")
    return y, notes


def _resample(y: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return y.astype(np.float32, copy=False)
    divisor = math.gcd(int(source_sr), int(target_sr))
    up = target_sr // divisor
    down = source_sr // divisor
    resampled = signal.resample_poly(y, up, down).astype(np.float32)
    return resampled


def _spectral_gate(y: np.ndarray) -> np.ndarray:
    """Subtract a noise floor estimated from the quietest frames."""
    nperseg = 512
    noverlap = 256
    if len(y) < nperseg:
        return y
    _, _, spectrum = signal.stft(y, fs=ANALYSIS_SR, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(spectrum)
    power = magnitude.mean(axis=0)
    quiet_count = max(1, int(0.12 * power.size))
    quiet_index = np.argpartition(power, quiet_count - 1)[:quiet_count]
    noise = np.median(magnitude[:, quiet_index], axis=1, keepdims=True)
    cleaned = np.maximum(magnitude - 0.85 * noise, 0.12 * magnitude)
    rebuilt = cleaned * np.exp(1j * np.angle(spectrum))
    _, output = signal.istft(rebuilt, fs=ANALYSIS_SR, nperseg=nperseg, noverlap=noverlap)
    if len(output) < len(y):
        padded = np.zeros(len(y), dtype=np.float32)
        padded[: len(output)] = output[: len(y)]
        return padded
    return output[: len(y)].astype(np.float32)


def _rms_normalize(y: np.ndarray, target: float = 0.07) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y * y) + 1e-12))
    if rms < 1e-6:
        raise ValueError("Audio is silent after preprocessing.")
    y = y * (target / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.98:
        y = y * (0.98 / peak)
    return y.astype(np.float32)


def _fade(y: np.ndarray, sample_rate: int, milliseconds: float = 8.0) -> np.ndarray:
    ramp_len = min(len(y) // 4, int(sample_rate * milliseconds / 1000))
    if ramp_len < 2:
        return y
    ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
    y = y.copy()
    y[:ramp_len] *= ramp
    y[-ramp_len:] *= ramp[::-1]
    return y
