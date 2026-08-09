from __future__ import annotations

from dataclasses import replace
from enum import StrEnum

from lib.parser.model import MaimaiChart, MaimaiNote


class Rotation(StrEnum):
    """The four rotational symmetries currently used for augmentation."""

    IDENTITY = "Identity"
    CLOCKWISE_90 = "Clockwise90"
    CLOCKWISE_180 = "Clockwise180"
    COUNTERCLOCKWISE_90 = "Counterclockwise90"


_POSITION_OFFSETS: dict[Rotation, int] = {
    Rotation.IDENTITY: 0,
    Rotation.CLOCKWISE_90: 2,
    Rotation.CLOCKWISE_180: 4,
    Rotation.COUNTERCLOCKWISE_90: -2,
}


def _rotation(value: Rotation | str) -> Rotation:
    try:
        return Rotation(value)
    except ValueError as error:
        supported = ", ".join(rotation.value for rotation in Rotation)
        raise ValueError(
            f"unsupported chart rotation {value!r}; expected one of: {supported}"
        ) from error


def rotate_position(
    position: int,
    rotation: Rotation | str,
    *,
    touch_area: str | None = None,
) -> int:
    """Rotate a one-based maimai position.

    Outer-ring and non-centre touch positions use the same eight-position
    rotation.  The centre C touch is fixed at universal position 8.
    """
    rotation = _rotation(rotation)
    if not 1 <= position <= 8:
        raise ValueError(f"maimai position must be in 1..8, got {position}")
    if touch_area == "C":
        # The centre panel carries no direction; its placeholder key stays put.
        return position
    offset = _POSITION_OFFSETS[rotation]
    return ((position - 1 + offset) % 8) + 1


def rotate_note(note: MaimaiNote, rotation: Rotation | str) -> MaimaiNote:
    """Return a rotated copy of one physical note."""
    rotation = _rotation(rotation)
    key_id = rotate_position(note.key_id, rotation, touch_area=note.key_group or None)
    if note.slide_info is None:
        return replace(note, key_id=key_id)
    return replace(
        note,
        key_id=key_id,
        slide_info=replace(
            note.slide_info,
            start_position=key_id,
            end_position=rotate_position(
                note.slide_info.end_position,
                rotation,
                touch_area=note.key_group or None,
            ),
        ),
    )


def rotate_chart(chart: MaimaiChart, rotation: Rotation | str) -> MaimaiChart:
    """Return an independent spatially rotated chart.

    All semantic timing and note flags are preserved.
    """
    rotation = _rotation(rotation)
    return replace(
        chart,
        notes=[rotate_note(note, rotation) for note in chart.notes],
        warnings=list(chart.warnings),
    )


def rotation_variants(
    chart: MaimaiChart,
    rotations: list[Rotation | str],
) -> dict[Rotation, MaimaiChart]:
    """Create unique named rotation variants from one parsed chart."""
    normalized = [_rotation(rotation) for rotation in rotations]
    if len(set(normalized)) != len(normalized):
        raise ValueError("rotations must not contain duplicates")
    return {rotation: rotate_chart(chart, rotation) for rotation in normalized}


__all__ = [
    "Rotation",
    "rotate_chart",
    "rotate_note",
    "rotate_position",
    "rotation_variants",
]
