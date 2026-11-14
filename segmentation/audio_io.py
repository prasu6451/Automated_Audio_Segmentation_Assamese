"""Read and write PCM WAV files, including 24-bit studio recordings."""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np


def read_wav(source) -> tuple[np.ndarray, int, dict]:
    """Load a PCM WAV as mono float32 in [-1, 1].

    `source` is a filesystem path or a seekable binary file object.
    Stereo channels are averaged. 16-bit, 24-bit, and 32-bit PCM are supported.
    """
    opened = str(source) if not hasattr(source, "read") else source
    with wave.open(opened, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        comptype = wf.getcomptype()
        raw = wf.readframes(nframes)

    if sample_rate < 8000 or sample_rate > 192000:
        raise ValueError(f"Unsupported sample rate: {sample_rate} Hz.")
    if channels < 1 or channels > 2:
        raise ValueError(f"Expected mono or stereo audio, found {channels} channels.")
    if nframes < 1 or len(raw) < channels * sampwidth:
        raise ValueError("The WAV file has no audio samples.")

    samples = _decode_pcm(raw, sampwidth, nframes * channels)
    if channels > 1:
        usable = (len(samples) // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    info = {
        "channels": channels,
        "sampwidth": sampwidth,
        "sample_rate": sample_rate,
        "nframes": int(len(samples)),
        "duration": float(len(samples) / sample_rate),
    }
    return samples.astype(np.float32, copy=False), sample_rate, info


def write_wav(destination, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono float audio as 16-bit PCM WAV."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    opened = destination if hasattr(destination, "write") else os.fspath(destination)
    with wave.open(opened, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _decode_pcm(raw: bytes, sampwidth: int, n_samples: int) -> np.ndarray:
    needed = n_samples * sampwidth
    raw = raw[:needed]
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return data / 2147483648.0
    if sampwidth == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8)
        if len(bytes_) < n_samples * 3:
            n_samples = len(bytes_) // 3
        triples = bytes_[: n_samples * 3].reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values >= 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    raise ValueError(f"Unsupported sample width: {sampwidth * 8}-bit.")


def safe_wav_name(folder: Path, filename: str) -> Path:
    """Resolve a dataset filename and reject paths that leave the folder."""
    if not filename or Path(filename).name != filename:
        raise ValueError("Invalid audio filename.")
    if Path(filename).suffix.lower() != ".wav":
        raise ValueError("Dataset files must be WAV.")
    root = folder.resolve()
    candidate = (root / filename).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise ValueError("Audio file was not found in the dataset.")
    return candidate
