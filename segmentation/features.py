"""Frame-level acoustic features: energy, ZCR, spectrum, pitch, and MFCC."""

from __future__ import annotations

import numpy as np

from segmentation.config import F0_MAX, F0_MIN, FRAME_S, HOP_S, N_MELS, N_MFCC, ANALYSIS_SR


def frame_signal(audio: np.ndarray, sample_rate: int = ANALYSIS_SR) -> tuple[np.ndarray, int, int]:
    """Return overlapping frames shaped (n_frames, frame_length)."""
    frame_len = int(round(FRAME_S * sample_rate))
    hop = int(round(HOP_S * sample_rate))
    if len(audio) < frame_len:
        raise ValueError("Audio is shorter than one analysis frame.")
    count = 1 + (len(audio) - frame_len) // hop
    index = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    return audio[index].astype(np.float32), frame_len, hop


def extract_features(audio: np.ndarray, sample_rate: int = ANALYSIS_SR) -> dict:
    frames, frame_len, hop = frame_signal(audio, sample_rate)
    window = np.hanning(frame_len).astype(np.float32)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32)
    power = magnitude * magnitude

    ste = np.mean(frames * frames, axis=1)
    signs = np.signbit(frames)
    zcr = np.mean(np.abs(np.diff(signs.astype(np.int8), axis=1)), axis=1).astype(np.float32)

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate).astype(np.float32)
    centroid = (magnitude * freqs).sum(axis=1) / (magnitude.sum(axis=1) + 1e-8)
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (np.mean(power, axis=1) + 1e-12)

    f0 = _pitch(windowed, sample_rate)
    mfcc = _mfcc(power, sample_rate, frame_len)
    times = (np.arange(len(ste)) * hop + frame_len / 2) / sample_rate

    return {
        "frames": len(ste),
        "frame_len": frame_len,
        "hop": hop,
        "times": times.astype(np.float32),
        "ste": ste.astype(np.float32),
        "zcr": zcr,
        "centroid": centroid.astype(np.float32),
        "flatness": flatness.astype(np.float32),
        "f0": f0,
        "mfcc": mfcc.astype(np.float32),
        "magnitude": magnitude,
        "freqs": freqs,
    }


def _pitch(windowed: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_len = windowed.shape[1]
    padded = int(2 ** np.ceil(np.log2(frame_len * 2)))
    spectrum = np.fft.rfft(windowed, n=padded, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), n=padded, axis=1)[:, :frame_len]
    min_lag = max(1, int(sample_rate / F0_MAX))
    max_lag = min(frame_len - 2, int(sample_rate / F0_MIN))
    if max_lag <= min_lag + 2:
        return np.full(len(windowed), np.nan, dtype=np.float32)

    region = autocorr[:, min_lag:max_lag]
    energy = autocorr[:, 0] + 1e-8
    relative = region / energy[:, None]
    peak_value = relative.max(axis=1)
    peak_lag = relative.argmax(axis=1) + min_lag

    f0 = np.full(len(windowed), np.nan, dtype=np.float32)
    voiced = peak_value > 0.42
    for index in np.flatnonzero(voiced):
        lag = int(peak_lag[index])
        left = relative[index, lag - min_lag - 1] if lag - min_lag - 1 >= 0 else relative[index, 0]
        center = relative[index, lag - min_lag]
        right = relative[index, lag - min_lag + 1] if lag - min_lag + 1 < region.shape[1] else center
        denom = left - 2 * center + right
        delta = 0.5 * (left - right) / denom if abs(denom) > 1e-6 else 0.0
        refined = lag + float(np.clip(delta, -0.5, 0.5))
        half = refined / 2
        if half >= min_lag:
            half_index = int(round(half)) - min_lag
            if 0 <= half_index < region.shape[1] and relative[index, half_index] > 0.9 * center:
                refined = half
        f0[index] = sample_rate / refined
    return f0


def _mfcc(power: np.ndarray, sample_rate: int, frame_len: int) -> np.ndarray:
    filters = _mel_filterbank(sample_rate, frame_len, N_MELS)
    mel_energy = power @ filters.T
    log_mel = np.log(mel_energy + 1e-10)
    basis = _dct_basis(N_MFCC, N_MELS)
    return log_mel @ basis.T


def _mel_filterbank(sample_rate: int, frame_len: int, n_mels: int) -> np.ndarray:
    n_freq = frame_len // 2 + 1

    def hz_to_mel(hz: float | np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(80.0), hz_to_mel(sample_rate / 2), n_mels + 2)
    bins = np.floor((frame_len + 1) * mel_to_hz(mels) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
    bank = np.zeros((n_mels, n_freq), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, min(center, n_freq)):
            bank[i, j] = (j - left) / (center - left)
        for j in range(center, min(right, n_freq)):
            bank[i, j] = (right - j) / (right - center)
    return bank


def _dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    basis = np.cos(np.pi * k * (n + 0.5) / n_mels).astype(np.float32)
    basis[0] *= np.sqrt(1.0 / n_mels)
    basis[1:] *= np.sqrt(2.0 / n_mels)
    return basis
