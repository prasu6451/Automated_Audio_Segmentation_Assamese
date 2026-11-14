"""Run preprocessing, VAD, features, and boundary detection as one pipeline."""

from __future__ import annotations

import base64
import time

import numpy as np

from segmentation.boundaries import boundary_times, detect_boundaries
from segmentation.config import ANALYSIS_SR
from segmentation.evaluate import evaluate_boundaries
from segmentation.features import extract_features
from segmentation.preprocess import prepare_audio
from segmentation.vad import detect_speech


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    denoise: bool = True,
    already_prepared: bool = False,
    vad_method: str = "energy_spectral",
    min_pause: float = 0.5,
    use_features: bool = True,
    target: str = "sentence",
    reference_times: list[float] | None = None,
    tolerance: float = 0.25,
    notes: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Segment speech and return prepared audio plus a JSON-ready payload."""
    log = list(notes or [])
    stages = []

    started = time.perf_counter()
    if already_prepared:
        prepared = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != ANALYSIS_SR:
            raise ValueError("Prepared audio must already be 16 kHz.")
        log.append("Using audio that was already preprocessed.")
    else:
        prepared, prep_notes = prepare_audio(audio, sample_rate, denoise=denoise)
        log.extend(prep_notes)
    duration = float(len(prepared) / ANALYSIS_SR)
    stages.append({"id": "preprocess", "label": "Audio preprocessing", "ms": _ms(started)})

    started = time.perf_counter()
    features = extract_features(prepared, ANALYSIS_SR)
    log.append(
        f"Framed the signal into {features['frames']} windows "
        f"of {int(features['frame_len'] / ANALYSIS_SR * 1000)} ms "
        f"with a {int(features['hop'] / ANALYSIS_SR * 1000)} ms hop."
    )
    mask, vad_info = detect_speech(features, vad_method if vad_method == "energy" else "energy_spectral")
    log.append(
        f"VAD ({vad_info['method']}): {vad_info['speech_ratio'] * 100:.0f}% of frames classified as speech."
    )
    stages.append({"id": "vad", "label": "Voice activity detection", "ms": _ms(started)})

    started = time.perf_counter()
    log.append("Extracted short-time energy, zero-crossing rate, pitch (F0), and 13 MFCCs.")
    spectrogram = _spectrogram(features["magnitude"])
    feature_plot = {
        "ste": _reduce(features["ste"], 480, voiced=False),
        "zcr": _reduce(features["zcr"], 480, voiced=False),
        "f0": _reduce(features["f0"], 480, voiced=True),
        "mfcc1": _reduce(features["mfcc"][:, 1], 480, voiced=False),
    }
    stages.append({"id": "features", "label": "Acoustic feature extraction", "ms": _ms(started)})

    started = time.perf_counter()
    segments = detect_boundaries(
        mask,
        features,
        duration,
        min_pause=min_pause,
        use_features=use_features,
        target=target if target == "word" else "sentence",
    )
    decision = "pause duration + acoustic features" if use_features else "pause duration only"
    unit = "word" if target == "word" else "sentence"
    log.append(
        f"Boundary decision ({decision}), minimum pause {min_pause:.2f} s: "
        f"{len(segments)} {unit} segment{'s' if len(segments) != 1 else ''}."
    )
    stages.append({"id": "boundaries", "label": "Boundary detection and segmentation", "ms": _ms(started)})

    pauses = _pauses(segments)
    metrics = None
    if reference_times is not None:
        metrics = evaluate_boundaries(boundary_times(segments), reference_times, tolerance)
        metrics["source"] = "reference"
        log.append(
            f"Scored {metrics['n_hyp']} detected boundaries against {metrics['n_ref']} "
            f"reference boundaries at ±{tolerance:.2f} s."
        )

    wave_min, wave_max = _envelope(prepared, 1000)
    payload = {
        "duration": round(duration, 4),
        "sample_rate": ANALYSIS_SR,
        "target": "word" if target == "word" else "sentence",
        "vad_method": vad_info["method"],
        "min_pause": min_pause,
        "use_features": use_features,
        "segments": segments,
        "waveform": {"min": wave_min, "max": wave_max},
        "spectrogram": spectrogram,
        "features": feature_plot,
        "stats": {
            "speech_ratio": round(vad_info["speech_ratio"], 4),
            "n_segments": len(segments),
            "mean_segment": round(float(np.mean([s["duration"] for s in segments])), 3),
            "mean_pause": round(float(np.mean(pauses)), 3) if pauses else 0.0,
            "n_pauses": len(pauses),
        },
        "metrics": metrics,
        "stages": stages,
        "log": log,
    }
    return prepared, payload


def _spectrogram(magnitude: np.ndarray) -> dict:
    # magnitude: frames x frequency. Display uses frequency x time, low frequency first.
    image = magnitude.T.astype(np.float32)
    n_freq = min(96, image.shape[0])
    n_time = min(220, image.shape[1])
    pooled = _pool(image, n_freq, n_time)
    db = 20.0 * np.log10(pooled + 1e-8)
    low, high = np.percentile(db, [5, 99])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((db - low) / (high - low), 0.0, 1.0)
    raw = (scaled * 255.0).astype(np.uint8)
    return {
        "n_freq": n_freq,
        "n_time": n_time,
        "f_max": ANALYSIS_SR / 2,
        "data_b64": base64.b64encode(raw.tobytes()).decode("ascii"),
    }


def _pool(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    height, width = image.shape
    row_edges = np.linspace(0, height, out_h + 1).astype(int)
    col_edges = np.linspace(0, width, out_w + 1).astype(int)
    pooled = np.empty((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        r0, r1 = int(row_edges[i]), max(int(row_edges[i + 1]), int(row_edges[i]) + 1)
        row = image[r0:r1]
        for j in range(out_w):
            c0, c1 = int(col_edges[j]), max(int(col_edges[j + 1]), int(col_edges[j]) + 1)
            pooled[i, j] = float(row[:, c0:c1].mean())
    return pooled


def _envelope(audio: np.ndarray, bins: int) -> tuple[list[float], list[float]]:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return [], []
    bins = min(bins, len(audio))
    usable = (len(audio) // bins) * bins
    shaped = audio[:usable].reshape(bins, -1)
    low = np.round(shaped.min(axis=1), 4)
    high = np.round(shaped.max(axis=1), 4)
    return low.astype(float).tolist(), high.astype(float).tolist()


def _reduce(values: np.ndarray, count: int, voiced: bool) -> list:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return []
    groups = np.array_split(np.arange(len(values)), min(count, len(values)))
    reduced = []
    for group in groups:
        chunk = values[group]
        if voiced:
            finite = chunk[np.isfinite(chunk)]
            reduced.append(None if len(finite) == 0 else round(float(np.median(finite)), 2))
        else:
            reduced.append(round(float(np.mean(chunk)), 5))
    return reduced


def _pauses(segments: list[dict]) -> list[float]:
    pauses = []
    for left, right in zip(segments, segments[1:]):
        gap = right["start"] - left["end"]
        if gap > 0:
            pauses.append(gap)
    return pauses


def _ms(started: float) -> int:
    return int(round((time.perf_counter() - started) * 1000))
