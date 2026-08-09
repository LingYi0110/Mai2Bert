#!/usr/bin/env python3

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from .model import MaimaiChart, MaimaiNote, MaimaiNoteType, SlideInfo

SLIDE_PATTERNS = {
    "SI_": "straight",
    "SCL": "circular_left",
    "SCR": "circular_right",
    "SV_": "circular_down",
    "SUL": "p_shape",
    "SUR": "q_shape",
    "SSL": "s_shape",
    "SSR": "z_shape",
    "SXL": "pp_shape",
    "SXR": "qq_shape",
    "SF_": "fan",
    "SLL": "v_fold_left",
    "SLR": "v_fold_right",
}

_SLIDE_TYPES = frozenset(SLIDE_PATTERNS)
_PREFIXES = frozenset({"NM", "BR", "EX", "BX", "CN"})
_STATE_FLAGS = {
    "NM": (False, False),
    "BR": (True, False),
    "EX": (False, True),
    "BX": (True, True),
    "CN": (False, False),
}
_LEGACY_TYPES = {
    "XTP": ("TAP", "EX"),
    "BRK": ("TAP", "BR"),
    "XST": ("STR", "EX"),
    "BST": ("STR", "BR"),
    "XHO": ("HLD", "EX"),
}


@dataclass(frozen=True, slots=True)
class _BpmEvent:
    tick: int
    bpm: float
    source_line: int


@dataclass(slots=True)
class _RawNote:
    base_type: str
    state: str
    tick: int
    start_key: int
    source_line: int
    length: int = 0
    wait: int = 0
    end_key: int | None = None
    touch_area: str | None = None
    special_effect: bool = False

    @property
    def is_break(self) -> bool:
        return _STATE_FLAGS[self.state][0]

    @property
    def is_ex(self) -> bool:
        return _STATE_FLAGS[self.state][1]


class _Timeline:
    """Piecewise MA2 tick-to-seconds conversion."""

    def __init__(self, events: list[_BpmEvent], resolution: int) -> None:
        if resolution <= 0:
            raise ValueError("RESOLUTION must be positive")
        if not events:
            raise ValueError("MA2 chart contains no BPM definition")

        by_tick: dict[int, _BpmEvent] = {}
        for event in events:
            if not math.isfinite(event.bpm) or event.bpm <= 0:
                raise ValueError(f"invalid BPM at line {event.source_line}: {event.bpm}")
            by_tick[event.tick] = event
        ordered = sorted(by_tick.values(), key=lambda event: event.tick)
        if ordered[0].tick > 0:
            ordered.insert(0, _BpmEvent(0, ordered[0].bpm, ordered[0].source_line))

        self.events = ordered
        self.resolution = resolution
        self.ticks = [event.tick for event in ordered]
        self.seconds: list[float] = [0.0]
        for previous, current in zip(ordered, ordered[1:], strict=False):
            elapsed = (current.tick - previous.tick) * self._seconds_per_tick(previous.bpm)
            self.seconds.append(self.seconds[-1] + elapsed)

    def _seconds_per_tick(self, bpm: float) -> float:
        return (60.0 / bpm) * (4.0 / self.resolution)

    def time_at(self, tick: int) -> float:
        index = max(bisect_right(self.ticks, tick) - 1, 0)
        event = self.events[index]
        return self.seconds[index] + (tick - event.tick) * self._seconds_per_tick(event.bpm)


class Ma2Parser:
    """Parse MA2 text into the simplified :class:`MaimaiChart` model."""

    def __init__(self, *, strict: bool = False, max_warnings: int = 200) -> None:
        self.strict = strict
        self.max_warnings = max_warnings

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        strict: bool = False,
    ) -> MaimaiChart:
        raw = Path(path).read_bytes()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        return cls(strict=strict).parse(content, chart_id=Path(path).stem)

    @classmethod
    def from_string(
        cls,
        content: str,
        *,
        chart_id: str = "",
        strict: bool = False,
    ) -> MaimaiChart:
        return cls(strict=strict).parse(content, chart_id=chart_id)

    def parse(self, content: str, *, chart_id: str = "") -> MaimaiChart:
        chart = MaimaiChart(chart_id=chart_id)
        lines = content.splitlines()
        resolution = self._read_resolution(lines, chart)
        bpm_events: list[_BpmEvent] = []
        raw_notes: list[_RawNote] = []

        for line_number, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("\t")
            command = fields[0]
            try:
                if command == "BPM_DEF":
                    self._require(fields, 5, line_number)
                    values = [self._float(value, "BPM_DEF") for value in fields[1:5]]
                    if any(value <= 0 for value in values):
                        raise ValueError("BPM_DEF values must be positive")
                elif command == "BPM":
                    self._require(fields, 4, line_number)
                    tick = self._absolute_tick(fields[1], fields[2], resolution)
                    bpm_events.append(_BpmEvent(tick, self._float(fields[3], "BPM"), line_number))
                else:
                    raw_note = self._parse_note(fields, line, line_number, resolution)
                    if raw_note is not None:
                        raw_notes.append(raw_note)
            except (IndexError, TypeError, ValueError) as error:
                self._handle_error(chart, line_number, line, error)

        try:
            timeline = _Timeline(bpm_events, resolution)
        except ValueError as error:
            if self.strict:
                raise
            chart.warnings.append(str(error))
            return chart

        notes = self._build_notes(raw_notes, timeline)
        notes.sort(key=lambda note: (note.start_time, note.source_index))
        chart.notes = notes
        return chart

    def _read_resolution(self, lines: list[str], chart: MaimaiChart) -> int:
        for line_number, raw_line in enumerate(lines, 1):
            fields = raw_line.strip().split("\t")
            if not fields or fields[0] != "RESOLUTION":
                continue
            try:
                self._require(fields, 2, line_number)
                resolution = int(fields[1])
                if resolution <= 0:
                    raise ValueError("RESOLUTION must be positive")
                return resolution
            except ValueError as error:
                if self.strict:
                    raise
                self._warn(chart, f"line {line_number}: {error}; using 384")
                return 384
        if self.strict:
            raise ValueError("MA2 chart contains no RESOLUTION header")
        self._warn(chart, "missing RESOLUTION; using 384")
        return 384

    def _parse_note(
        self,
        fields: list[str],
        source: str,
        line_number: int,
        resolution: int,
    ) -> _RawNote | None:
        raw_type = fields[0]
        state = "NM"
        base_type = raw_type
        if raw_type in _LEGACY_TYPES:
            base_type, state = _LEGACY_TYPES[raw_type]
        elif len(raw_type) == 5 and raw_type[:2] in _PREFIXES:
            state, base_type = raw_type[:2], raw_type[2:]

        recognized = base_type in {"TAP", "STR", "TTP", "HLD", "THO"} | _SLIDE_TYPES
        if not recognized:
            return None
        if state == "CN" and base_type not in _SLIDE_TYPES:
            raise ValueError(f"CN prefix is only valid for slide segments: {raw_type}")
        self._require(fields, 4, line_number)
        tick = self._absolute_tick(fields[1], fields[2], resolution)
        key = self._key(fields[3])

        note = _RawNote(base_type, state, tick, key, line_number, source)
        if base_type == "TTP":
            self._require(fields, 6, line_number)
            note.touch_area = self._touch_area(fields[4])
            note.special_effect = int(fields[5]) == 1
        elif base_type == "THO":
            self._require(fields, 7, line_number)
            note.length = self._nonnegative_int(fields[4], "touch-hold length")
            note.touch_area = self._touch_area(fields[5])
            note.special_effect = int(fields[6]) == 1
        elif base_type == "HLD":
            self._require(fields, 5, line_number)
            note.length = self._nonnegative_int(fields[4], "hold length")
        elif base_type in _SLIDE_TYPES:
            self._require(fields, 7, line_number)
            note.wait = self._nonnegative_int(fields[4], "slide wait")
            note.length = self._nonnegative_int(fields[5], "slide duration")
            note.end_key = self._key(fields[6])
        return note

    def _build_notes(self, raw_notes: list[_RawNote], timeline: _Timeline) -> list[MaimaiNote]:
        result: list[MaimaiNote] = []
        for raw in raw_notes:
            if raw.base_type == "STR":
                result.append(self._tap_note(raw, timeline))
            elif raw.base_type in _SLIDE_TYPES:
                result.append(self._slide_note(raw, timeline))
            else:
                result.append(self._ordinary_note(raw, timeline))
        return result

    def _ordinary_note(self, raw: _RawNote, timeline: _Timeline) -> MaimaiNote:
        note_type = {
            "HLD": MaimaiNoteType.HOLD,
            "THO": MaimaiNoteType.TOUCH_HOLD,
            "TTP": MaimaiNoteType.TOUCH,
        }.get(raw.base_type, MaimaiNoteType.TAP)
        start_time = timeline.time_at(raw.tick)
        duration = (
            timeline.time_at(raw.tick + raw.length) - start_time
            if note_type in {MaimaiNoteType.HOLD, MaimaiNoteType.TOUCH_HOLD}
            else 0.0
        )
        return MaimaiNote(
            source_index=raw.source_line,
            start_time=start_time,
            duration=duration,
            type=note_type,
            key_id=raw.start_key + 1,
            key_group=raw.touch_area or "",
            is_break=raw.is_break,
            is_fireworks=raw.special_effect,
            is_ex=raw.is_ex,
        )

    def _tap_note(self, raw: _RawNote, timeline: _Timeline) -> MaimaiNote:
        return MaimaiNote(
            source_index=raw.source_line,
            start_time=timeline.time_at(raw.tick),
            duration=0.0,
            type=MaimaiNoteType.TAP,
            key_id=raw.start_key + 1,
            is_break=raw.is_break,
            is_ex=raw.is_ex,
        )

    def _slide_note(self, raw: _RawNote, timeline: _Timeline) -> MaimaiNote:
        assert raw.end_key is not None
        start_time = timeline.time_at(raw.tick)
        delay = timeline.time_at(raw.tick + raw.wait) - start_time
        duration = timeline.time_at(raw.tick + raw.wait + raw.length) - start_time
        return MaimaiNote(
            source_index=raw.source_line,
            start_time=start_time,
            duration=duration,
            type=MaimaiNoteType.SLIDE,
            key_id=raw.start_key + 1,
            is_break=raw.is_break,
            is_ex=raw.is_ex,
            slide_info=SlideInfo(
                pattern=SLIDE_PATTERNS[raw.base_type],
                start_position=raw.start_key + 1,
                end_position=raw.end_key + 1,
                duration=duration,
                delay=delay,
            ),
        )

    @staticmethod
    def _absolute_tick(bar: str, tick: str, resolution: int) -> int:
        bar_value = int(bar)
        tick_value = int(tick)
        if bar_value < 0 or tick_value < 0:
            raise ValueError("MA2 bar and tick must be non-negative")
        return bar_value * resolution + tick_value

    @staticmethod
    def _key(value: str) -> int:
        key = int(value)
        if not 0 <= key <= 7:
            raise ValueError(f"MA2 key out of range: {key}")
        return key

    @staticmethod
    def _touch_area(value: str) -> str:
        if value not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"invalid touch area: {value!r}")
        return value

    @staticmethod
    def _nonnegative_int(value: str, field: str) -> int:
        result = int(value)
        if result < 0:
            raise ValueError(f"{field} must be non-negative")
        return result

    @staticmethod
    def _float(value: str, field: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{field} must be finite")
        return result

    def _require(self, fields: list[str], count: int, line_number: int) -> None:
        if len(fields) < count:
            raise ValueError(
                f"line {line_number}: expected at least {count} fields, got {len(fields)}"
            )

    def _handle_error(
        self, chart: MaimaiChart, line_number: int, line: str, error: Exception
    ) -> None:
        message = f"line {line_number}: {line[:60]!r}: {error}"
        if self.strict:
            raise ValueError(message)
        self._warn(chart, message)

    def _warn(self, chart: MaimaiChart, message: str) -> None:
        if len(chart.warnings) < self.max_warnings:
            chart.warnings.append(message)
