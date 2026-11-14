"""Pause- and feature-based sentence or word boundary detection."""

from __future__ import annotations

import numpy as np

from segmentation.config import HOP_S


def detect_boundaries(
    speech_mask: np.ndarray,
    features: dict,
    duration: float,
    min_pause: float,
    use_features: bool,
    target: str,
) -> list[dict]:
    hop = HOP_S
    if target == "word":
        fill_frames = max(1, int(round(min(0.05, min_pause * 0.4) / hop)))
        min_speech_frames = max(1, int(round(0.06 / hop)))
        min_segment = 0.10
    else:
        fill_frames = max(1, int(round(min(0.20, min_pause * 0.45) / hop)))
        min_speech_frames = max(1, int(round(0.10 / hop)))
        min_segment = 0.35

    mask = _remove_short_runs(speech_mask, min_speech_frames, keep_true=False)
    mask = _fill_short_gaps(mask, fill_frames)
    mask = _remove_short_runs(mask, min_speech_frames, keep_true=False)
    regions = _runs(mask, hop, duration)
    if not regions:
        return [_segment(1, 0.0, duration, None)]

    merged: list[list[float]] = [list(regions[0])]
    confidences: list[float | None] = []
    for start, end in regions[1:]:
        pause_start = merged[-1][1]
        pause_dur = start - pause_start
        score = _pause_score(features, pause_start, start, min_pause) if pause_dur > 0 else 0.0
        accept = pause_dur + 1e-6 >= min_pause and (
            not use_features or pause_dur >= min_pause * 1.65 or score >= 0.45
        )
        if accept:
            confidences.append(round(score, 3))
            merged.append([start, end])
        else:
            merged[-1][1] = end

    confidences.append(None)
    merged = _absorb_short_segments(merged, min_segment, confidences)
    padded = _pad(merged, duration, pad=0.04)
    segments = []
    for index, ((start, end), confidence) in enumerate(zip(padded, confidences), start=1):
        segments.append(_segment(index, start, end, confidence))
    return segments


def boundary_times(segments: list[dict]) -> list[float]:
    """Midpoint of the gap between consecutive segments. One time per internal boundary."""
    times = []
    for left, right in zip(segments, segments[1:]):
        times.append(round((left["end"] + right["start"]) / 2, 4))
    return times


def _segment(index: int, start: float, end: float, confidence: float | None) -> dict:
    start = round(max(0.0, start), 4)
    end = round(max(start + 0.01, end), 4)
    return {
        "id": f"S{index}",
        "index": index,
        "start": start,
        "end": end,
        "duration": round(end - start, 4),
        "transcript": "",
        "boundary_confidence": confidence,
    }


def _runs(mask: np.ndarray, hop: float, duration: float) -> list[tuple[float, float]]:
    regions = []
    index = 0
    n = len(mask)
    while index < n:
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < n and mask[end]:
            end += 1
        t0 = index * hop
        t1 = min(duration, end * hop)
        if end == n and mask[-1]:
            t1 = duration
        if t1 - t0 > 0.02:
            regions.append((t0, t1))
        index = end
    return regions


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.copy()
    index = 0
    n = len(mask)
    while index < n:
        if mask[index]:
            index += 1
            continue
        end = index
        while end < n and not mask[end]:
            end += 1
        has_left = index > 0 and mask[index - 1]
        has_right = end < n and mask[end]
        if has_left and has_right and (end - index) <= max_gap:
            mask[index:end] = True
        index = end
    return mask


def _remove_short_runs(mask: np.ndarray, min_len: int, keep_true: bool) -> np.ndarray:
    """Drop short True runs. `keep_true` is unused and kept so call sites read clearly."""
    del keep_true
    mask = mask.copy()
    index = 0
    n = len(mask)
    while index < n:
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < n and mask[end]:
            end += 1
        if (end - index) < min_len:
            mask[index:end] = False
        index = end
    return mask


def _pause_score(features: dict, pause_start: float, pause_end: float, min_pause: float) -> float:
    times = features["times"]
    if len(times) == 0 or pause_end <= pause_start:
        return 0.0
    pause = (times >= pause_start) & (times < pause_end)
    pre = (times >= pause_start - 0.18) & (times < pause_start)
    post = (times >= pause_end) & (times < pause_end + 0.18)
    if not np.any(pause):
        return 0.0

    ste = features["ste"]
    speech_energy = 0.0
    speech_parts = []
    if np.any(pre):
        speech_parts.append(float(np.mean(ste[pre])))
    if np.any(post):
        speech_parts.append(float(np.mean(ste[post])))
    if speech_parts:
        speech_energy = max(speech_parts)
    pause_energy = float(np.mean(ste[pause]))
    energy_drop = float(np.clip(1.0 - pause_energy / (speech_energy + 1e-8), 0.0, 1.0))

    pause_dur = pause_end - pause_start
    pause_score = float(np.clip(pause_dur / (min_pause * 2), 0.0, 1.0))

    f0 = features["f0"]
    pitch_reset = 0.35
    pre_f0 = f0[pre]
    post_f0 = f0[post]
    pre_f0 = pre_f0[np.isfinite(pre_f0)]
    post_f0 = post_f0[np.isfinite(post_f0)]
    if len(pre_f0) and len(post_f0):
        ratio = float(np.median(post_f0) / max(float(np.median(pre_f0)), 1e-3))
        pitch_reset = float(np.clip(abs(np.log(max(ratio, 1e-3))) / np.log(1.45), 0.0, 1.0))

    flat = features["flatness"]
    flat_rise = 0.0
    if speech_parts:
        speech_flat_parts = []
        if np.any(pre):
            speech_flat_parts.append(float(np.mean(flat[pre])))
        if np.any(post):
            speech_flat_parts.append(float(np.mean(flat[post])))
        speech_flat = float(np.mean(speech_flat_parts))
        flat_rise = float(np.clip((float(np.mean(flat[pause])) - speech_flat) / 0.35, 0.0, 1.0))

    return 0.34 * pause_score + 0.36 * energy_drop + 0.15 * pitch_reset + 0.15 * flat_rise


def _absorb_short_segments(segments: list[list[float]], min_segment: float, confidences: list) -> list[list[float]]:
    guard = 0
    while len(segments) > 1 and guard < len(segments) + 2:
        guard += 1
        durations = [end - start for start, end in segments]
        short = next((i for i, dur in enumerate(durations) if dur < min_segment), None)
        if short is None:
            break
        if short == 0:
            _merge_into(segments, confidences, 0, 1)
        else:
            _merge_into(segments, confidences, short - 1, short)
    return segments


def _merge_into(segments, confidences, keep: int, drop: int) -> None:
    segments[keep][0] = min(segments[keep][0], segments[drop][0])
    segments[keep][1] = max(segments[keep][1], segments[drop][1])
    # The boundary that disappears is the one that belonged to the earlier segment.
    earlier = min(keep, drop)
    if earlier < len(confidences) - 1:
        confidences.pop(earlier)
    segments.pop(drop)


def _pad(segments: list[list[float]], duration: float, pad: float = 0.04) -> list[tuple[float, float]]:
    padded = []
    for index, (start, end) in enumerate(segments):
        left_limit = 0.0 if index == 0 else (segments[index - 1][1] + start) / 2
        right_limit = duration if index == len(segments) - 1 else (end + segments[index + 1][0]) / 2
        padded.append((round(left_limit, 4), round(right_limit, 4)))
    return padded
