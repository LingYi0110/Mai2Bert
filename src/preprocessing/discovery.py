from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lib.parser.model import (
    MaimaiChart,
    MaimaiNote,
    MaimaiNoteType,
    SlideInfo,
)

INDEX_HEADER = ("file", "label_type", "label")
ChartKey = tuple[str, str]


class LabelValue(float):
    label_type: Literal["precise", "coarse"]

    def __new__(cls, value: float, label_type: Literal["precise", "coarse"]) -> LabelValue:
        instance = float.__new__(cls, value)
        instance.label_type = label_type
        return instance

    def __reduce__(self) -> tuple[type[LabelValue], tuple[float, Literal["precise", "coarse"]]]:
        return LabelValue, (float(self), self.label_type)

    @property
    def difficulty_const(self) -> float:
        return float(self)


@dataclass(frozen=True, slots=True)
class ChartRecord:
    music_id: str
    path: Path
    dataset: str
    format: str
    difficulty_const: float | None = None
    label_type: Literal["precise", "coarse"] | None = None

    @property
    def key(self) -> ChartKey:
        return self.dataset, self.music_id


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    datasets: tuple[str, ...]
    charts_by_dataset: dict[str, int]
    standard_charts: int
    labeled_charts: int
    unlabeled_charts: int


def _coarse_to_float(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+)(\+)?\s*", value)
    if match is None:
        raise ValueError(f"invalid coarse difficulty label: {value!r}")
    return float(match.group(1)) + (0.5 if match.group(2) else 0.0)


def _index_target(row: dict[str, str], *, source: Path, line_number: int) -> LabelValue | None:
    label_type = row.get("label_type", "").strip()
    label = row.get("label", "").strip()
    if not label_type:
        return None
    if label_type == "precise":
        target = float(label)
        if not math.isfinite(target):
            raise ValueError(f"non-finite precise label at {source}:{line_number}")
        return LabelValue(target, "precise")
    if label_type == "coarse":
        return LabelValue(_coarse_to_float(label), "coarse")
    raise ValueError(f"unsupported label_type at {source}:{line_number}: {label_type!r}")


def load_dataset_labels(path: str | Path) -> dict[str, LabelValue | None]:
    source = Path(path)
    labels: dict[str, LabelValue | None] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = tuple(reader.fieldnames or ())
        if header != INDEX_HEADER:
            raise ValueError(f"{source} header must be {INDEX_HEADER!r}")
        for line_number, row in enumerate(reader, start=2):
            filename = row.get("file", "").strip()
            if not filename:
                raise ValueError(f"empty file name at {source}:{line_number}")
            if filename in labels:
                raise ValueError(f"duplicate file entry at {source}:{line_number}: {filename}")
            labels[filename] = _index_target(row, source=source, line_number=line_number)
    return labels


def _float(value: object, field: str, *, default: float = 0.0) -> float:
    result = default if value is None else float(str(value))
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _int(value: object, field: str, *, default: int = 0) -> int:
    return default if value is None else int(str(value))


def _bool(value: object) -> bool:
    return bool(value)


def _load_slide_info(value: object) -> SlideInfo | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("slide_info must be an object or null")
    return SlideInfo(
        pattern=str(value.get("pattern", "")),
        start_position=_int(value.get("start_position"), "slide.start_position"),
        end_position=_int(value.get("end_position"), "slide.end_position"),
        duration=_float(value.get("duration"), "slide.duration"),
        delay=_float(value.get("delay"), "slide.delay"),
    )


def _load_note(value: object, index: int) -> MaimaiNote:
    if not isinstance(value, dict):
        raise ValueError("notes entries must be objects")
    note_type = MaimaiNoteType(str(value.get("type")))
    return MaimaiNote(
        source_index=_int(value.get("source_index"), "note.source_index", default=index),
        start_time=_float(value.get("start_time"), "note.start_time"),
        duration=_float(value.get("duration"), "note.duration"),
        type=note_type,
        key_id=_int(value.get("key_id"), "note.key_id"),
        key_group=str(value.get("key_group", "")),
        is_break=_bool(value.get("is_break")),
        is_fireworks=_bool(value.get("is_fireworks")),
        is_ex=_bool(value.get("is_ex")),
        slide_info=_load_slide_info(value.get("slide_info")),
    )


def load_json_chart(path: str | Path) -> MaimaiChart:
    """Rebuild the simplified model from one exported chart JSON file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"chart JSON root must be an object: {source}")
    raw_notes = value.get("notes", [])
    if not isinstance(raw_notes, list):
        raise ValueError(f"chart JSON notes must be a list: {source}")
    notes = [_load_note(item, index) for index, item in enumerate(raw_notes)]
    return MaimaiChart(
        chart_id=str(value.get("chart_id", source.stem)),
        notes=notes,
        warnings=[str(item) for item in value.get("warnings", [])]
        if isinstance(value.get("warnings", []), list)
        else [],
    )


def discover_json_corpus(
    datasets_root: str | Path,
    datasets: list[str],
) -> tuple[list[ChartRecord], DiscoveryReport]:
    root = Path(datasets_root)
    if not root.is_dir():
        raise FileNotFoundError(f"datasets root does not exist: {root}")

    charts: list[ChartRecord] = []
    charts_by_dataset: dict[str, int] = {}
    for dataset in datasets:
        dataset_root = root / dataset
        index_path = dataset_root / "index.csv"
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
        if not index_path.is_file():
            raise FileNotFoundError(f"dataset index does not exist: {index_path}")
        labels = load_dataset_labels(index_path)
        for filename, difficulty_const in labels.items():
            path = dataset_root / filename
            if not path.is_file():
                raise ValueError(f"indexed chart file does not exist: {path}")
            charts.append(
                ChartRecord(
                    music_id=f"{dataset}:{filename}",
                    path=path.resolve(),
                    difficulty_const=difficulty_const,
                    label_type=difficulty_const.label_type
                    if difficulty_const is not None
                    else None,
                    dataset=dataset,
                    format="json",
                )
            )
        charts_by_dataset[dataset] = len(labels)

    charts.sort(key=lambda chart: (chart.dataset, chart.music_id))
    report = DiscoveryReport(
        datasets=tuple(datasets),
        charts_by_dataset=charts_by_dataset,
        standard_charts=len(charts),
        labeled_charts=sum(chart.difficulty_const is not None for chart in charts),
        unlabeled_charts=sum(chart.difficulty_const is None for chart in charts),
    )
    return charts, report
