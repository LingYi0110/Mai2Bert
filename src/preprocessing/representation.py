from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.parser.model import MaimaiChart, MaimaiNote, MaimaiNoteType

SPECIAL_TOKENS = {"PAD": 0, "CLS": 1, "MASK": 2, "NA": 3, "UNK": 4}

CATEGORICAL_FIELDS = (
    "note_type",
    "key_id",
    "key_group",
    "is_break",
    "is_fireworks",
    "is_ex",
    "slide_pattern",
    "slide_end",
)
CONTINUOUS_FIELDS = ("delta_seconds", "delay_seconds", "duration_seconds")

# One (mean, std) pair per continuous channel, in CONTINUOUS_FIELDS order.
CONTINUOUS_NORMALIZATION = (
    (0.06621158, 0.07203260),
    (0.11925023, 0.05443141),
    (0.12431587, 0.07168549),
)

NOTE_TYPES = tuple(note_type.value for note_type in MaimaiNoteType)
POSITIONS = tuple(str(position) for position in range(1, 9))
TOUCH_AREAS = ("A", "B", "C", "D", "E")
SLIDE_SHAPES = (
    "straight",
    "circular_left",
    "circular_right",
    "circular_up",
    "circular_down",
    "v_fold_left",
    "v_fold_right",
    "p_shape",
    "q_shape",
    "pp_shape",
    "qq_shape",
    "s_shape",
    "z_shape",
    "fan",
)
FLAG_FIELDS = ("is_break", "is_fireworks", "is_ex")


def _vocabulary(values: tuple[str, ...]) -> dict[str, int]:
    return {value: index + len(SPECIAL_TOKENS) for index, value in enumerate(values)}


VOCABULARIES: dict[str, dict[str, int]] = {
    "note_type": _vocabulary(NOTE_TYPES),
    "key_id": _vocabulary(POSITIONS),
    "key_group": _vocabulary(TOUCH_AREAS),
    **{field: _vocabulary(("False", "True")) for field in FLAG_FIELDS},
    "slide_pattern": _vocabulary(SLIDE_SHAPES),
    "slide_end": _vocabulary(POSITIONS),
}
VOCAB_SIZES = tuple(max(vocabulary.values()) + 1 for vocabulary in VOCABULARIES.values())
assert tuple(VOCABULARIES) == CATEGORICAL_FIELDS, "vocabulary order must match field order"


def representation_schema() -> dict[str, Any]:
    """Return the complete, hashable v3 representation contract."""
    return {
        "special_tokens": SPECIAL_TOKENS,
        "categorical_fields": CATEGORICAL_FIELDS,
        "continuous_fields": CONTINUOUS_FIELDS,
        "vocabularies": VOCABULARIES,
        "ordering": ["timing_seconds", "source_index"],
        "continuous_transform": {
            "delta_seconds": {"operation": "log1p", "scale": "log(11)", "clip": "[0, clip]"},
            "delay_seconds": {"operation": "log1p", "scale": "log(11)", "clip": "[0, clip]"},
            "duration_seconds": {
                "operation": "log1p",
                "scale": "log(31)",
                "clip": "[0, clip]",
            },
        },
        "presence_grammar": {
            "NOTE": {
                "categorical_required": [
                    "note_type",
                    "key_id",
                    *FLAG_FIELDS,
                ],
                "key_group": "required exactly for Touch and TouchHold",
                "slide_pattern": "required exactly for Slide",
                "slide_end": "required exactly for Slide",
                "delta_seconds": "absent on first physical note, required thereafter",
                "delay_seconds": "required exactly for Slide",
                "duration_seconds": "required exactly for Hold, TouchHold, and Slide",
            },
        },
    }


def schema_hash(schema: Mapping[str, Any] | None = None) -> str:
    """Hash every part of a representation schema using canonical JSON."""
    value = json.dumps(
        representation_schema() if schema is None else schema,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EventArrays:
    """Dense event rows plus explicit per-feature presence."""

    categorical: NDArray[np.int16]
    categorical_presence: NDArray[np.bool_]
    continuous: NDArray[np.float32]
    continuous_presence: NDArray[np.bool_]
    note_start: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.categorical.ndim != 2 or self.categorical_presence.ndim != 2:
            raise ValueError("categorical arrays must be two-dimensional")
        if self.continuous.ndim != 2 or self.continuous_presence.ndim != 2:
            raise ValueError("continuous arrays must be two-dimensional")
        if self.categorical.shape != self.categorical_presence.shape:
            raise ValueError("categorical values and presence must have equal shapes")
        if self.continuous.shape != self.continuous_presence.shape:
            raise ValueError("continuous values and presence must have equal shapes")
        if self.categorical.shape[0] != self.continuous.shape[0]:
            raise ValueError("categorical and continuous arrays must have equal lengths")
        if self.note_start.ndim != 1 or self.note_start.shape[0] != self.categorical.shape[0]:
            raise ValueError("note_start must be one-dimensional with one value per row")
        if self.categorical.dtype != np.dtype(np.int16):
            raise TypeError("categorical must have dtype int16")
        if self.categorical_presence.dtype != np.dtype(np.bool_):
            raise TypeError("categorical_presence must have dtype bool")
        if self.continuous.dtype != np.dtype(np.float32):
            raise TypeError("continuous must have dtype float32")
        if self.continuous_presence.dtype != np.dtype(np.bool_):
            raise TypeError("continuous_presence must have dtype bool")
        if self.note_start.dtype != np.dtype(np.bool_):
            raise TypeError("note_start must have dtype bool")
        if not np.isfinite(self.continuous).all():
            raise ValueError("continuous values must be finite")

    @property
    def length(self) -> int:
        return int(self.categorical.shape[0])

    @property
    def note_count(self) -> int:
        return int(np.count_nonzero(self.note_start))


def _validate_position(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
        raise ValueError(f"{label} must be an integer in 1..8, got {value!r}")


def _validate_nonnegative_finite(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")


def _validate_note(note: MaimaiNote) -> None:
    if not isinstance(note.type, MaimaiNoteType):
        raise ValueError(f"unsupported note_type {note.type!r}")
    _validate_position(note.key_id, "note key_id")
    _validate_nonnegative_finite(note.start_time, "note start_time")
    _validate_nonnegative_finite(note.duration, "note duration")
    for field in FLAG_FIELDS:
        if not isinstance(getattr(note, field), bool):
            raise ValueError(f"{field} must be bool")

    is_touch = note.type in {MaimaiNoteType.TOUCH, MaimaiNoteType.TOUCH_HOLD}
    if is_touch:
        if note.key_group not in TOUCH_AREAS:
            raise ValueError(f"touch note key_group must be one of A-E, got {note.key_group!r}")
    elif note.key_group:
        raise ValueError("key_group is only valid for Touch and TouchHold notes")

    if note.type is MaimaiNoteType.SLIDE:
        info = note.slide_info
        if info is None:
            raise ValueError("slide note must carry slide_info")
        if info.pattern not in SLIDE_SHAPES:
            raise ValueError(f"unsupported slide pattern {info.pattern!r}")
        _validate_position(info.start_position, "slide start_position")
        _validate_position(info.end_position, "slide end_position")
        _validate_nonnegative_finite(info.delay, "slide delay")
        _validate_nonnegative_finite(info.duration, "slide duration")
        if info.start_position != note.key_id:
            raise ValueError(
                f"slide start_position {info.start_position} must equal key_id {note.key_id}"
            )
    elif note.slide_info is not None:
        raise ValueError("slide_info is only valid for Slide notes")


def _category(field: str, value: object) -> int:
    return VOCABULARIES[field].get(str(value), SPECIAL_TOKENS["UNK"])


def _set_category(
    values: NDArray[np.int16],
    presence: NDArray[np.bool_],
    row: int,
    field: str,
    value: object,
) -> None:
    column = CATEGORICAL_FIELDS.index(field)
    values[row, column] = _category(field, value)
    presence[row, column] = True


def _transform(value: float, scale: float, clip: float) -> np.float32:
    return np.float32(np.clip(math.log1p(value) / math.log(scale), 0.0, clip))


def chart_to_arrays(chart: MaimaiChart, *, clip: float = 10.0) -> EventArrays:
    """Validate and expand a parsed chart into v3 NOTE rows."""
    if not isinstance(chart, MaimaiChart):
        raise TypeError("chart_to_arrays expects a MaimaiChart")
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip must be finite and positive")

    notes = sorted(chart.notes, key=lambda note: (note.start_time, note.source_index))
    for note in notes:
        _validate_note(note)

    length = len(notes)
    categorical = np.full((length, len(CATEGORICAL_FIELDS)), SPECIAL_TOKENS["NA"], dtype=np.int16)
    categorical_presence = np.zeros_like(categorical, dtype=np.bool_)
    continuous = np.zeros((length, len(CONTINUOUS_FIELDS)), dtype=np.float32)
    continuous_presence = np.zeros_like(continuous, dtype=np.bool_)
    note_start = np.ones(length, dtype=np.bool_)

    previous_seconds: float | None = None
    for row, note in enumerate(notes):
        _set_category(categorical, categorical_presence, row, "note_type", note.type.value)
        _set_category(categorical, categorical_presence, row, "key_id", note.key_id)
        if note.key_group:
            _set_category(categorical, categorical_presence, row, "key_group", note.key_group)
        for field in FLAG_FIELDS:
            _set_category(categorical, categorical_presence, row, field, bool(getattr(note, field)))

        if note.type is MaimaiNoteType.SLIDE:
            assert note.slide_info is not None
            _set_category(
                categorical,
                categorical_presence,
                row,
                "slide_pattern",
                note.slide_info.pattern,
            )
            _set_category(
                categorical, categorical_presence, row, "slide_end", note.slide_info.end_position
            )

        if previous_seconds is not None:
            delta = max(note.start_time - previous_seconds, 0.0)
            continuous[row, 0] = _transform(delta, 11.0, clip)
            continuous_presence[row, 0] = True
        if note.type is MaimaiNoteType.SLIDE:
            assert note.slide_info is not None
            continuous[row, 1] = _transform(note.slide_info.delay, 11.0, clip)
            continuous_presence[row, 1] = True
        if note.type in {
            MaimaiNoteType.HOLD,
            MaimaiNoteType.TOUCH_HOLD,
            MaimaiNoteType.SLIDE,
        }:
            continuous[row, 2] = _transform(note.duration, 31.0, clip)
            continuous_presence[row, 2] = True

        previous_seconds = note.start_time

    return EventArrays(
        categorical=categorical,
        categorical_presence=categorical_presence,
        continuous=continuous,
        continuous_presence=continuous_presence,
        note_start=note_start,
    )
