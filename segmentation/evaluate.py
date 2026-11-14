"""Boundary precision, recall, F1, and R-value."""

from __future__ import annotations

import math

from segmentation.boundaries import boundary_times


def evaluate_boundaries(hypothesis: list[float], reference: list[float], tolerance: float) -> dict:
    """Score internal boundary times with a one-to-one tolerance collar.

    R-value follows Räsänen, Laine, and Altosaar (2009): 1 is a perfect
    segmentation and lower values mix misses with over-segmentation.
    """
    hyp = sorted(float(t) for t in hypothesis)
    ref = sorted(float(t) for t in reference)
    tolerance = max(0.0, float(tolerance))
    used: set[int] = set()
    hits = 0
    for ref_time in ref:
        best_index = None
        best_distance = tolerance
        for index, hyp_time in enumerate(hyp):
            if index in used:
                continue
            distance = abs(hyp_time - ref_time)
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            used.add(best_index)
            hits += 1

    n_ref = len(ref)
    n_hyp = len(hyp)
    precision = hits / n_hyp if n_hyp else (1.0 if n_ref == 0 else 0.0)
    recall = hits / n_ref if n_ref else (1.0 if n_hyp == 0 else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if n_ref == 0 and n_hyp == 0:
        r_value = 1.0
    elif n_ref == 0 or precision == 0:
        r_value = 0.0
    else:
        overseg = recall / precision - 1.0
        r1 = math.sqrt((1.0 - recall) ** 2 + overseg ** 2)
        r2 = (-overseg + recall - 1.0) / math.sqrt(2.0)
        r_value = 1.0 - (abs(r1) + abs(r2)) / 2.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "r_value": round(max(-1.0, min(1.0, r_value)), 4),
        "hits": hits,
        "n_ref": n_ref,
        "n_hyp": n_hyp,
        "tolerance": tolerance,
    }


def evaluate_segments(hyp_segments: list[dict], ref_segments: list[dict], tolerance: float) -> dict:
    return evaluate_boundaries(
        boundary_times(hyp_segments),
        boundary_times(ref_segments),
        tolerance,
    )
