from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MaimaiNoteType(StrEnum):
    """Format-neutral physical note kinds."""

    TAP = "Tap"
    SLIDE = "Slide"
    HOLD = "Hold"
    TOUCH = "Touch"
    TOUCH_HOLD = "TouchHold"


@dataclass(slots=True)
class SlideInfo:
    """Collapsed single-segment slide geometry.

    ``duration`` is the total span from slide start to the end of the last
    segment (for the official MA2 source this equals wait + length converted
    to seconds); ``delay`` is the wait before the slide body starts.  A
    headless continuation (MA2 ``CN`` rows, simai multi-segment splits, ``!``
    no-head syntax) always carries ``delay == 0``, which is what distinguishes
    it from a headed slide.
    """

    pattern: str
    start_position: int
    end_position: int
    duration: float
    delay: float


@dataclass(slots=True)
class MaimaiNote:
    """One physical maimai note with seconds as its time axis."""

    start_time: float
    duration: float
    type: MaimaiNoteType
    key_id: int
    key_group: str = ""
    is_break: bool = False
    is_fireworks: bool = False
    is_ex: bool = False
    slide_info: SlideInfo | None = None
    source_index: int = 0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass(slots=True)
class MaimaiChart:
    """One playable difficulty of a chart.

    Difficulty is not stored in the chart file; it belongs in the dataset
    ``index.csv`` label column.
    """

    chart_id: str
    notes: list[MaimaiNote] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the simplified chart JSON layout."""
        return {
            "chart_id": self.chart_id,
            "notes": [asdict(note) for note in self.notes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MaimaiChart:
        """Deserialize the simplified chart JSON layout (inverse of ``to_dict``)."""
        notes: list[MaimaiNote] = []
        for raw in payload["notes"]:
            slide_raw = raw.get("slide_info")
            slide_info: SlideInfo | None = None
            if slide_raw is not None:
                slide_info = SlideInfo(
                    pattern=str(slide_raw["pattern"]),
                    start_position=int(slide_raw["start_position"]),
                    end_position=int(slide_raw["end_position"]),
                    duration=float(slide_raw["duration"]),
                    delay=float(slide_raw["delay"]),
                )
            notes.append(
                MaimaiNote(
                    start_time=float(raw["start_time"]),
                    duration=float(raw["duration"]),
                    type=MaimaiNoteType(raw["type"]),
                    key_id=int(raw["key_id"]),
                    key_group=str(raw.get("key_group", "")),
                    is_break=bool(raw.get("is_break", False)),
                    is_fireworks=bool(raw.get("is_fireworks", False)),
                    is_ex=bool(raw.get("is_ex", False)),
                    slide_info=slide_info,
                    source_index=int(raw.get("source_index", 0)),
                )
            )
        return cls(
            chart_id=str(payload["chart_id"]),
            notes=notes,
            warnings=list(payload.get("warnings", [])),
        )
