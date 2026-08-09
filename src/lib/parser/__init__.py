from pathlib import Path
from typing import Literal

from .ma2 import Ma2Parser
from .model import MaimaiChart, MaimaiNote, MaimaiNoteType, SlideInfo
from .simai import MaimaiFile, SimaiCommand, SimaiParser

ChartFormat = Literal["ma2", "simai"]


def parse_chart_file(
    path: str | Path,
    *,
    format: ChartFormat | None = None,
    difficulty: int | None = None,
    strict: bool = False,
) -> MaimaiChart:
    """Parse one MA2 file or one difficulty from a Simai maidata file."""
    source = Path(path)
    selected = format or ("simai" if source.suffix.lower() == ".txt" else "ma2")
    if selected == "ma2":
        if difficulty is not None:
            raise ValueError("difficulty is only valid for Simai charts")
        return Ma2Parser.from_file(source, strict=strict)
    if selected == "simai":
        if difficulty is None:
            raise ValueError("Simai chart parsing requires difficulty 1..7")
        result = SimaiParser.from_file(source, difficulty=difficulty, strict=strict)
        if not isinstance(result, MaimaiChart):
            raise TypeError("Simai parser did not return a chart")
        return result
    raise ValueError(f"unsupported chart format: {selected!r}")


__all__ = [
    "ChartFormat",
    "Ma2Parser",
    "MaimaiChart",
    "MaimaiFile",
    "MaimaiNote",
    "MaimaiNoteType",
    "SimaiCommand",
    "SimaiParser",
    "SlideInfo",
    "parse_chart_file",
]
