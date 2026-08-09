from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _rankdata(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _subset_metrics(
    predictions: NDArray[np.float64],
    targets: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "mae": math.nan, "bias": math.nan}
    errors = predictions[mask] - targets[mask]
    return {
        "count": count,
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
    }


def regression_metrics(
    predictions: list[float] | NDArray[np.float64],
    targets: list[float] | NDArray[np.float64],
    groups: list[str],
    *,
    threshold: float = 12.0,
    bin_width: float = 0.1,
) -> dict[str, Any]:
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("predictions and targets must be same-length vectors")
    if len(groups) != len(target):
        raise ValueError("groups must align with predictions")
    if len(target) == 0:
        raise ValueError("cannot compute regression metrics for an empty dataset")
    errors = prediction - target
    absolute = np.abs(errors)
    centered = target - target.mean()
    denominator = float(np.square(centered).sum())

    metrics: dict[str, Any] = {
        "count": len(target),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "r2": float(1.0 - np.square(errors).sum() / denominator) if denominator > 0 else math.nan,
        "pearson": _correlation(prediction, target),
        "spearman": _correlation(_rankdata(prediction), _rankdata(target)),
        "bias": float(errors.mean()),
    }
    for tolerance in (0.1, 0.2, 0.5, 1.0):
        metrics[f"within_{tolerance:g}"] = float(np.mean(absolute <= tolerance))
    for cutoff in (threshold, 14.0, 14.5, 15.0):
        name = f"ge_{cutoff:g}".replace(".", "_")
        metrics[name] = _subset_metrics(prediction, target, target >= cutoff)

    per_bin: dict[str, dict[str, float | int]] = {}
    bin_ids = np.floor((target + 1e-10) / bin_width).astype(np.int64)
    high_bins = sorted(set(bin_ids[target >= threshold].tolist()))
    for bin_id in high_bins:
        mask = bin_ids == bin_id
        label = f"{bin_id * bin_width:.1f}"
        per_bin[label] = _subset_metrics(prediction, target, mask)
    metrics["per_high_bin"] = per_bin
    valid_bin_maes = [value["mae"] for value in per_bin.values() if value["count"]]
    metrics["high_macro_bin_mae"] = float(np.mean(valid_bin_maes)) if valid_bin_maes else math.nan

    group_errors: dict[str, list[float]] = defaultdict(list)
    for group, error in zip(groups, absolute.tolist(), strict=True):
        group_errors[group].append(error)
    metrics["canonical_song_macro_mae"] = float(
        np.mean([np.mean(values) for values in group_errors.values()])
    )
    return metrics


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)):
            flattened[name] = float(value)
    return flattened
