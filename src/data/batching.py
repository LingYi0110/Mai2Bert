from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

from preprocessing.representation import (
    CONTINUOUS_NORMALIZATION,
)

_CONT_MEANS = torch.tensor([m for m, _ in CONTINUOUS_NORMALIZATION], dtype=torch.float32)
_CONT_STDS = torch.tensor([s for _, s in CONTINUOUS_NORMALIZATION], dtype=torch.float32)


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        *,
        max_tokens: int,
        max_batch_size: int,
        bucket_size: int = 256,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """Length-bucketed batch sampler. """
        if not lengths:
            raise ValueError("lengths cannot be empty")
        if max_tokens <= 0 or max_batch_size <= 0 or bucket_size <= 0:
            raise ValueError("batch limits must be positive")
        self.lengths = list(lengths)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        buckets = [
            indices[start : start + self.bucket_size]
            for start in range(0, len(indices), self.bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=self.lengths.__getitem__)
        if self.shuffle:
            rng.shuffle(buckets)

        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        for index in (item for bucket in buckets for item in bucket):
            length = self.lengths[index]
            exceeds = current and (
                current_tokens + length > self.max_tokens or len(current) >= self.max_batch_size
            )
            if exceeds:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(index)
            current_tokens += length
            if length > self.max_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches()
        if self.shuffle:
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches())


def compound_window_ranges(
    note_start: Sequence[bool] | torch.Tensor,
    length: int,
    max_events: int,
    stride: int,
) -> list[tuple[int, int]]:
    """Return compound-aligned windows covering all ``length`` valid events.

    Each true value in ``note_start`` begins a NOTE compound. Windows start and
    end only at those boundaries (or at ``length``), so NOTE segment rows are
    never separated from their NOTE row.
    """
    if length < 0 or length > len(note_start):
        raise ValueError("length must be between zero and len(note_start)")
    if max_events <= 0 or stride <= 0:
        raise ValueError("max_events and stride must be positive")
    if length == 0:
        return []

    starts = [index for index in range(length) if bool(note_start[index])]
    if not starts or starts[0] != 0:
        raise ValueError("the first valid event must start a NOTE compound")
    boundaries = [*starts, length]
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end - start > max_events:
            raise ValueError(
                f"NOTE compound [{start}, {end}) has {end - start} events, "
                f"exceeding max_events={max_events}"
            )

    ranges: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(starts):
        start = starts[start_index]
        end_index = start_index + 1
        while end_index < len(boundaries) and boundaries[end_index] - start <= max_events:
            end_index += 1
        end_index -= 1
        end = boundaries[end_index]
        ranges.append((start, end))
        if end == length:
            break

        target = start + stride
        next_index = start_index + 1
        while next_index < len(starts) and starts[next_index] < target:
            next_index += 1
        # Target beyond window: advance one compound and let windows overlap.
        if next_index >= len(starts) or starts[next_index] > end:
            next_index = start_index + 1
        start_index = next_index

    return ranges


def collate_events(
    items: list[dict[str, Any]],
    *,
    pad_to_multiple: int = 1,
) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    maximum = max(int(item["categorical"].shape[0]) for item in items)
    padded_length = math.ceil(maximum / pad_to_multiple) * pad_to_multiple
    batch_size = len(items)
    categorical_fields = int(items[0]["categorical"].shape[1])
    continuous_fields = int(items[0]["continuous"].shape[1])
    categorical = torch.zeros((batch_size, padded_length, categorical_fields), dtype=torch.long)
    cat_presence = torch.zeros((batch_size, padded_length, categorical_fields), dtype=torch.bool)
    continuous = torch.zeros((batch_size, padded_length, continuous_fields), dtype=torch.float32)
    cont_presence = torch.zeros((batch_size, padded_length, continuous_fields), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.bool)
    note_start = torch.zeros((batch_size, padded_length), dtype=torch.bool)

    for row, item in enumerate(items):
        length = int(item["categorical"].shape[0])
        categorical[row, :length] = item["categorical"].long()
        cat_presence[row, :length] = item["categorical_presence"].bool()
        continuous[row, :length] = item["continuous"].float()
        cont_presence[row, :length] = item["continuous_presence"].bool()
        attention_mask[row, :length] = True
        if "note_start" not in item:
            raise KeyError("each event item must contain note_start")
        note_start[row, :length] = item["note_start"].bool()

    # Z-score present values; absent slots stay zero (flagged by continuous_presence).
    continuous = torch.where(
        cont_presence,
        (continuous - _CONT_MEANS) / _CONT_STDS,
        torch.zeros((), dtype=continuous.dtype),
    )

    result: dict[str, Any] = {
        "categorical": categorical,
        "categorical_presence": cat_presence,
        "continuous": continuous,
        "continuous_presence": cont_presence,
        "attention_mask": attention_mask,
        "note_start": note_start,
        "lengths": [int(item["categorical"].shape[0]) for item in items],
    }
    if "target" in items[0]:
        result["target"] = torch.stack([item["target"] for item in items])
    for key in ("coarse_min", "coarse_max"):
        if key in items[0]:
            result[key] = torch.stack([item[key] for item in items])
    if "is_coarse" in items[0]:
        result["is_coarse"] = torch.stack([item["is_coarse"] for item in items])
    if "window_ranges" in items[0]:
        result["window_ranges"] = [item["window_ranges"] for item in items]
    for key in ("dataset", "music_id", "rotation"):
        if key in items[0]:
            result[key] = [item[key] for item in items]
    return result
