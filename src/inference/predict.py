from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from data.batching import collate_events, compound_window_ranges
from lib.config import AppConfig
from lib.parser import ChartFormat, MaimaiChart, parse_chart_file
from preprocessing.representation import chart_to_arrays
from training.regression import RegressionModule


@dataclass(frozen=True, slots=True)
class PredictionResult:
    source: str
    format: ChartFormat
    difficulty: int | None
    predicted: float
    sigma: float
    n_notes: int
    n_windows: int
    warnings: tuple[str, ...] = ()


def _chart_item(
    chart: MaimaiChart,
    config: AppConfig,
) -> dict[str, torch.Tensor]:
    arrays = chart_to_arrays(chart, clip=config.representation.continuous_clip)
    return {
        "categorical": torch.from_numpy(arrays.categorical),
        "categorical_presence": torch.from_numpy(arrays.categorical_presence),
        "continuous": torch.from_numpy(arrays.continuous),
        "continuous_presence": torch.from_numpy(arrays.continuous_presence),
        "note_start": torch.from_numpy(arrays.note_start),
        "music_id": "",
        "rotation": "Identity",
    }


@torch.no_grad()
def predict_chart(
    module: RegressionModule,
    chart: MaimaiChart,
    config: AppConfig,
) -> tuple[float, float]:
    item = _chart_item(chart, config)
    batch = collate_events([item], pad_to_multiple=config.batch.pad_to_multiple)
    batch = {
        key: value.to(module.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    mean, sigma = module.predict_gaussian(batch)
    return float(mean.item()), float(sigma.item())


def predict_file(
    module: RegressionModule,
    config: AppConfig,
    path: str | Path,
    *,
    format: ChartFormat | None = None,
    difficulty: int | None = None,
) -> PredictionResult:
    source = Path(path)
    chart = parse_chart_file(source, format=format, difficulty=difficulty)
    mean, sigma = predict_chart(module, chart, config)
    arrays = chart_to_arrays(chart, clip=config.representation.continuous_clip)
    window_count = len(
        compound_window_ranges(
            arrays.note_start.tolist(),
            arrays.length,
            config.representation.max_events,
            config.representation.stride,
        )
    )
    return PredictionResult(
        source=str(source),
        format=format or ("simai" if source.suffix.lower() == ".txt" else "ma2"),
        difficulty=difficulty,
        predicted=mean,
        sigma=sigma,
        n_notes=len(chart.notes),
        n_windows=window_count,
    )


@torch.no_grad()
def predict_items(
    module: RegressionModule,
    items: list[dict[str, torch.Tensor]],
    config: AppConfig,
    *,
    batch_size: int = 32,
) -> list[tuple[float, float]]:
    results: list[tuple[float, float]] = [None] * len(items)  # type: ignore[list-item]
    order = sorted(
        range(len(items)),
        key=lambda index: int(items[index]["categorical"].shape[0]),
    )
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        batch = collate_events(
            [items[index] for index in chunk],
            pad_to_multiple=config.batch.pad_to_multiple,
        )
        batch = {
            key: value.to(module.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        mean, sigma = module.predict_gaussian(batch)
        for position, index in enumerate(chunk):
            results[index] = (float(mean[position].item()), float(sigma[position].item()))
    return results
