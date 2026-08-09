from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.batching import compound_window_ranges
from preprocessing.store import ProcessedStore, StoredVariant


class PretrainingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: ProcessedStore,
        *,
        max_events: int = 512,
        seed: int = 42,
    ) -> None:
        self.store = store
        self.max_events = max_events
        self.seed = seed
        self.variants = [variant for variant in store.variants if variant.pretraining]
        # Window RNGs live in the worker process so persistent workers re-draw windows each epoch.
        self._window_rngs: list[np.random.Generator | None] = [None] * len(self.variants)

    def __len__(self) -> int:
        return len(self.variants)

    @property
    def lengths(self) -> list[int]:
        return [min(variant.length, self.max_events) for variant in self.variants]

    def set_epoch(self, epoch: int) -> None:  # pragma: no cover - retained for API compatibility
        """Deprecated: window advancement is automatic via per-variant RNGs."""
        # Deliberate no-op; kept so external callers/tests do not break.
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        variant = self.variants[index]
        arrays = self.store.read(variant.row)
        length = variant.length
        if length > self.max_events:
            ranges = compound_window_ranges(
                torch.from_numpy(arrays["note_start"]),
                length,
                self.max_events,
                stride=1,
            )
            rng = self._window_rngs[index]
            if rng is None:
                rng = np.random.default_rng(self.seed + index)
                self._window_rngs[index] = rng
            start, end = ranges[int(rng.integers(len(ranges)))]
            selection = slice(start, end)
            arrays = {name: value[selection] for name, value in arrays.items()}
        return {
            **{name: torch.from_numpy(value) for name, value in arrays.items()},
            "dataset": variant.dataset,
            "music_id": variant.music_id,
            "rotation": variant.rotation,
        }


class RegressionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: ProcessedStore,
        split: Literal["train", "validation", "test"],
        *,
        label_type: Literal["all", "precise", "coarse"] = "all",
        coarse_label_ranges: dict[str, tuple[float, float | None]] | None = None,
        max_events: int = 512,
        stride: int = 384,
    ) -> None:
        self.store = store
        self.split = split
        self.max_events = max_events
        self.stride = stride
        self.coarse_label_ranges = coarse_label_ranges or {}
        candidates = [
            variant
            for variant in store.variants
            if variant.split == split
            and variant.difficulty_const is not None
            and variant.rotation == "Identity"
        ]
        if label_type == "all":
            self.variants = candidates
        else:
            if any(variant.label_type is None for variant in candidates):
                raise ValueError(
                    "label_type metadata is missing from the processed store; "
                    "rerun prepare before filtering supervised labels"
                )
            self.variants = [variant for variant in candidates if variant.label_type == label_type]
        if split == "train" and self.coarse_label_ranges:
            self.variants = [
                variant
                for variant in self.variants
                if variant.label_type != "coarse"
                or variant.dataset == "thirdparty"
                or self._coarse_range(float(variant.difficulty_const)) is not None
            ]
        if not self.variants:
            raise ValueError(f"no {label_type} supervised labels found in {split} split")

    def _coarse_range(self, label_value: float) -> tuple[float, float | None] | None:
        return self.coarse_label_ranges.get(f"{label_value:g}")

    def __len__(self) -> int:
        return len(self.variants)

    @property
    def lengths(self) -> list[int]:
        return [variant.length for variant in self.variants]

    @property
    def targets(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.asarray([variant.difficulty_const for variant in self.variants], dtype=np.float32)

    @property
    def groups(self) -> list[str]:
        return [variant.music_id for variant in self.variants]

    def __getitem__(self, index: int) -> dict[str, Any]:
        variant: StoredVariant = self.variants[index]
        arrays = self.store.read(variant.row)
        window_ranges = compound_window_ranges(
            arrays["note_start"].tolist(),
            variant.length,
            self.max_events,
            self.stride,
        )
        result = {
            **{name: torch.from_numpy(value) for name, value in arrays.items()},
            "window_ranges": window_ranges,
            "target": torch.tensor(variant.difficulty_const, dtype=torch.float32),
            "dataset": variant.dataset,
            "music_id": variant.music_id,
            "rotation": variant.rotation,
        }
        if variant.label_type == "coarse":
            label_range = self._coarse_range(float(variant.difficulty_const))
            if label_range is None:
                raise ValueError(
                    f"coarse label {variant.difficulty_const:g} is missing from coarse_label_ranges"
                )
            minimum, maximum = label_range
            result["coarse_min"] = torch.tensor(minimum, dtype=torch.float32)
            result["coarse_max"] = torch.tensor(
                maximum if maximum is not None else float("inf"),
                dtype=torch.float32,
            )
        else:
            result["coarse_min"] = torch.tensor(
                float(variant.difficulty_const), dtype=torch.float32
            )
            result["coarse_max"] = torch.tensor(
                float(variant.difficulty_const), dtype=torch.float32
            )
        result["is_coarse"] = torch.tensor(variant.label_type == "coarse", dtype=torch.bool)
        return result
