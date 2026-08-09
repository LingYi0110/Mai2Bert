#!/usr/bin/env python3

from __future__ import annotations

import base64
import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .model import MaimaiChart, MaimaiNote, MaimaiNoteType, SlideInfo

SLIDE_MARKS = ("qq", "pp", "-", "^", "v", "<", ">", "V", "p", "q", "s", "z", "w")
SLIDE_CHARACTERS = frozenset("-^v<>Vpqszw")
TOUCH_AREAS = frozenset("ABCDE")
DURATION_RE = re.compile(r"\[([^\]]*)\]")

SLIDE_PATTERN_MAP = {
    "-": "straight",
    "<": "circular_left",
    ">": "circular_right",
    "^": "circular_up",
    "v": "circular_down",
    "p": "p_shape",
    "q": "q_shape",
    "pp": "pp_shape",
    "qq": "qq_shape",
    "s": "s_shape",
    "z": "z_shape",
    "w": "fan",
}


@dataclass(frozen=True, slots=True)
class SimaiCommand:
    """Unknown maidata metadata, preserving source order and duplicates."""

    prefix: str
    value: str


@dataclass(slots=True)
class MaimaiFile:
    """A Simai maidata file, containing up to seven difficulty charts."""

    title: str = ""
    artist: str = ""
    offset_seconds: float = 0.0
    final_designer: str = ""
    source_hash: str = ""
    source_format: str = "simai"
    charts: dict[int, MaimaiChart] = field(default_factory=dict)
    commands: list[SimaiCommand] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def notes(self) -> list[MaimaiNote]:
        return [
            note for difficulty in sorted(self.charts) for note in self.charts[difficulty].notes
        ]

    def get_chart(self, difficulty: int) -> MaimaiChart:
        try:
            return self.charts[difficulty]
        except KeyError as error:
            available = ", ".join(map(str, sorted(self.charts))) or "none"
            raise KeyError(f"difficulty {difficulty} is absent; available: {available}") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "artist": self.artist,
            "offset_seconds": self.offset_seconds,
            "final_designer": self.final_designer,
            "source_hash": self.source_hash,
            "source_format": self.source_format,
            "commands": [asdict(command) for command in self.commands],
            "charts": {str(key): chart.to_dict() for key, chart in sorted(self.charts.items())},
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class _Metadata:
    title: str = ""
    artist: str = ""
    offset_seconds: float = 0.0
    final_designer: str = ""
    designers: dict[int, str] = field(default_factory=dict)
    levels: dict[int, str] = field(default_factory=dict)
    charts: dict[int, str] = field(default_factory=dict)
    commands: list[SimaiCommand] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _ParserState:
    time_seconds: float = 0.0
    bpm: float = 0.0
    division: float = 4.0
    cell_index: int = 0


@dataclass(slots=True)
class _NoteFlags:
    is_touch: bool = False
    is_break: bool = False
    is_hold: bool = False
    is_slide: bool = False
    is_slide_no_head: bool = False
    is_slide_no_head_and_delay: bool = False
    is_ex: bool = False
    is_hanabi: bool = False
    is_mine: bool = False


class SimaiParser:
    """Parse Simai text into MajSimai-compatible, seconds-based objects."""

    def __init__(self, *, strict: bool = False, max_warnings: int = 200) -> None:
        self.strict = strict
        self.max_warnings = max_warnings
        self._source_index = 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        difficulty: int | None = None,
        strict: bool = False,
    ) -> MaimaiFile | MaimaiChart:
        raw = Path(path).read_bytes()
        content = cls.decode(raw)
        digest = base64.b64encode(hashlib.md5(raw).digest()).decode("ascii")
        return cls(strict=strict).parse(
            content,
            difficulty=difficulty,
            source_hash=digest,
            source_name=Path(path).stem,
        )

    @classmethod
    def from_string(
        cls,
        content: str,
        *,
        difficulty: int | None = None,
        strict: bool = False,
    ) -> MaimaiFile | MaimaiChart:
        return cls(strict=strict).parse(content, difficulty=difficulty)

    @staticmethod
    def decode(raw: bytes) -> str:
        """Decode UTF-8/UTF-16 Simai files, honoring byte-order marks."""
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            return raw.decode("utf-16")
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw.decode("utf-8-sig")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            if raw.count(b"\x00") > len(raw) // 4:
                return raw.decode("utf-16-le")
            return raw.decode("utf-8", errors="replace")

    def parse(
        self,
        content: str,
        *,
        difficulty: int | None = None,
        source_hash: str | None = None,
        source_name: str = "",
    ) -> MaimaiFile | MaimaiChart:
        """Parse complete maidata text.

        Without ``difficulty`` a :class:`MaimaiFile` is returned.  Supplying a
        difficulty from 1 through 7 returns that :class:`MaimaiChart`.
        """
        content = content.lstrip("\ufeff")
        metadata = self._parse_metadata(content)
        digest = source_hash or base64.b64encode(
            hashlib.md5(content.encode("utf-8")).digest()
        ).decode("ascii")
        result = MaimaiFile(
            title=metadata.title,
            artist=metadata.artist,
            offset_seconds=metadata.offset_seconds,
            final_designer=metadata.final_designer,
            source_hash=digest,
            commands=metadata.commands,
            warnings=metadata.warnings,
        )

        requested = [difficulty] if difficulty is not None else sorted(metadata.charts)
        for chart_index in requested:
            if chart_index not in metadata.charts:
                continue
            self._source_index = 0
            try:
                chart = self.parse_chart(
                    metadata.charts[chart_index],
                    chart_id=(
                        f"{source_name}_{chart_index}" if source_name else f"inote_{chart_index}"
                    ),
                )
            except Exception as error:
                if self.strict:
                    raise
                chart = MaimaiChart(
                    chart_id=f"inote_{chart_index}",
                    warnings=[f"chart parse failed: {type(error).__name__}: {error}"],
                )
            result.charts[chart_index] = chart

        if difficulty is not None:
            return result.get_chart(difficulty)
        return result

    def parse_chart(
        self,
        fumen: str,
        *,
        chart_id: str = "",
    ) -> MaimaiChart:
        """Parse one fumen with MajSimai's source-ordered character scanner."""
        self._source_index = 0
        chart = MaimaiChart(chart_id=chart_id)
        if not fumen:
            return chart

        state = _ParserState()
        note_buffer: list[str] = []
        have_note = False
        index = 0

        while index < len(fumen):
            char = fumen[index]
            if char in "\r\n":
                index += 1
                continue

            if char == "|":
                if index + 1 >= len(fumen) or fumen[index + 1] != "|":
                    message = "unexpected single '|' character"
                    if self.strict:
                        raise ValueError(message)
                    self._warn(chart, state, message)
                    index += 1
                    continue
                line_end = fumen.find("\n", index + 2)
                if line_end == -1:
                    line_end = len(fumen)
                index = line_end
                continue

            if char == "(":
                have_note = False
                note_buffer.clear()
                end = fumen.find(")", index + 1)
                if end == -1:
                    message = "unclosed BPM directive"
                    if self.strict:
                        raise ValueError(message)
                    self._warn(chart, state, message)
                    break
                raw_bpm = "".join(fumen[index + 1 : end].split())
                try:
                    state.bpm = self._number(raw_bpm)
                except ValueError as error:
                    if self.strict:
                        raise
                    self._warn(chart, state, f"ignored BPM ({raw_bpm}): {error}")
                index = end + 1
                continue

            if char == "{":
                have_note = False
                note_buffer.clear()
                end = fumen.find("}", index + 1)
                if end == -1:
                    message = "unclosed grid division"
                    if self.strict:
                        raise ValueError(message)
                    self._warn(chart, state, message)
                    break
                raw_division = "".join(fumen[index + 1 : end].split())
                try:
                    if not raw_division:
                        raise ValueError("grid division is empty")
                    if raw_division.startswith("#"):
                        interval = self._number(raw_division[1:])
                        if state.bpm == 0 or interval == 0:
                            raise ValueError("absolute grid requires non-zero BPM and interval")
                        state.division = 240.0 / (state.bpm * interval)
                    else:
                        state.division = self._number(raw_division)
                except (ValueError, ZeroDivisionError) as error:
                    if self.strict:
                        raise
                    self._warn(chart, state, f"ignored division {{{raw_division}}}: {error}")
                index = end + 1
                continue

            if char == "<" and not have_note:
                end = fumen.find(">", index + 1)
                if end == -1:
                    message = "unclosed HS/SV directive"
                    if self.strict:
                        raise ValueError(message)
                    self._warn(chart, state, message)
                    break
                index = end + 1
                continue

            if char == ",":
                if have_note:
                    raw_note = "".join(note_buffer)
                    if raw_note:
                        self._parse_cell_notes(raw_note, state, chart)
                if state.bpm != 0 and state.division != 0:
                    state.time_seconds += (60.0 / state.bpm) * (4.0 / state.division)
                else:
                    self._warn(chart, state, "time did not advance because BPM or division is zero")
                state.cell_index += 1
                have_note = False
                note_buffer.clear()
                index += 1
                continue

            if not have_note and (char in "0123456789" or char in TOUCH_AREAS or char in "!?"):
                have_note = True
                note_buffer.clear()
            if have_note and not char.isspace():
                note_buffer.append(char)
            index += 1

        return chart

    def _parse_metadata(self, content: str) -> _Metadata:
        result = _Metadata()
        lines = content.splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            index += 1
            if not line.startswith("&"):
                continue
            if "=" not in line:
                result.warnings.append(f"ignored metadata without '=': {line[:80]}")
                continue

            key, value = line[1:].split("=", 1)
            if not key:
                continue
            inote_match = re.fullmatch(r"inote_([1-7])", key)
            if inote_match:
                chart_index = int(inote_match.group(1))
                body = [value]
                while index < len(lines) and not lines[index].lstrip().startswith("&"):
                    body.append(lines[index].strip())
                    index += 1
                result.charts[chart_index] = "\n".join(body).strip()
                continue

            if key == "title":
                if value and not result.title:
                    result.title = value
            elif key == "artist":
                if value and not result.artist:
                    result.artist = value
            elif key == "des":
                if value and not result.final_designer:
                    result.final_designer = value
            elif key == "first":
                try:
                    offset = float(value or 0)
                    if not math.isfinite(offset):
                        raise ValueError
                    result.offset_seconds = offset
                except ValueError:
                    result.offset_seconds = 0.0
                    result.warnings.append(f"invalid &first value: {value!r}")
            elif match := re.fullmatch(r"des_([1-7])", key):
                chart_index = int(match.group(1))
                if value and not result.designers.get(chart_index):
                    result.designers[chart_index] = value
            elif match := re.fullmatch(r"lv_([1-7])", key):
                chart_index = int(match.group(1))
                if value and not result.levels.get(chart_index):
                    result.levels[chart_index] = value
            else:
                result.commands.append(SimaiCommand(key, value))

        if not result.final_designer:
            for chart_index in range(1, 8):
                if result.designers.get(chart_index):
                    result.final_designer = result.designers[chart_index]
        return result

    def _parse_cell_notes(self, text: str, state: _ParserState, chart: MaimaiChart) -> None:
        """Parse one comma cell, including MajSimai pseudo-each timing."""
        base_time = state.time_seconds
        step = 1.875 / state.bpm if state.bpm > 0 else 0.0
        for group_index, raw_group in enumerate(part for part in text.split("`") if part):
            timing = base_time + group_index * step
            group = raw_group.strip()
            notes: list[MaimaiNote] = []
            compact_pair = len(group) == 2 and group.isdigit()
            tokens = [group] if compact_pair else group.split("/")
            for token in tokens:
                token = token.strip()
                if not token or token == "E":
                    continue
                try:
                    notes.extend(
                        self._parse_note_content(
                            token,
                            timing,
                            state,
                            chart,
                            expand_compact_pair=compact_pair,
                        )
                    )
                except Exception as error:
                    if self.strict:
                        raise
                    self._warn(
                        chart,
                        state,
                        f"skipped note {token!r}: {type(error).__name__}: {error}",
                    )
            chart.notes.extend(notes)

    def _parse_note_content(
        self,
        content: str,
        timing: float,
        state: _ParserState,
        chart: MaimaiChart,
        *,
        expand_compact_pair: bool,
    ) -> list[MaimaiNote]:
        if expand_compact_pair:
            result: list[MaimaiNote] = []
            for digit in content:
                result.extend(self._parse_single_note(digit, timing, state, chart))
            return result

        if "*" not in content:
            return self._parse_single_note(content, timing, state, chart)

        parts = [part for part in content.split("*") if part]
        if not parts:
            return []
        result: list[MaimaiNote] = []
        head_character = parts[0][0]
        for part_index, part in enumerate(parts):
            if part_index == 0:
                notes = self._parse_single_note(part, timing, state, chart)
            else:
                expanded = f"{head_character}{part}"
                notes = self._parse_single_note(expanded, timing, state, chart)
                for note in notes:
                    if note.slide_info is not None:
                        note.slide_info.delay = 0.0
            result.extend(notes)
        return result

    def _parse_single_note(
        self,
        token: str,
        timing: float,
        state: _ParserState,
        chart: MaimaiChart,
    ) -> list[MaimaiNote]:
        cleaned, flags = self._detect_flags(token)
        if not cleaned:
            return []
        if flags.is_mine and not flags.is_slide:
            return []

        if flags.is_touch:
            touch_area = cleaned[0]
            if touch_area not in TOUCH_AREAS:
                raise ValueError("touch area must be A..E")
            if touch_area == "C":
                key_id = 1
            elif len(cleaned) >= 2 and cleaned[1] in "12345678":
                key_id = int(cleaned[1])
            else:
                raise ValueError("non-C touch is missing its position")
            note_type = MaimaiNoteType.TOUCH_HOLD if flags.is_hold else MaimaiNoteType.TOUCH
            key_group = touch_area
        else:
            if cleaned[0] not in "12345678":
                raise ValueError("note is missing a 1..8 start position")
            key_id = int(cleaned[0])
            key_group = ""
            note_type = MaimaiNoteType.HOLD if flags.is_hold else MaimaiNoteType.TAP

        if flags.is_slide:
            note_type = MaimaiNoteType.SLIDE

        duration = 0.0
        if note_type in {
            MaimaiNoteType.HOLD,
            MaimaiNoteType.TOUCH_HOLD,
        } and (note_type is MaimaiNoteType.TOUCH_HOLD or not cleaned.endswith("h")):
            try:
                duration = self._hold_time(state.bpm, cleaned)
            except (ValueError, ZeroDivisionError) as error:
                self._warn(chart, state, f"hold duration fallback for {token!r}: {error}")
                duration = 0.0

        slide_info: SlideInfo | None = None
        if note_type is MaimaiNoteType.SLIDE:
            wait, total = self._slide_times(state.bpm, cleaned)
            segments = self._extract_slide_segments(cleaned, key_id)
            if not segments:
                raise ValueError("slide contains no geometric segment")
            delay = 0.0 if flags.is_slide_no_head and not flags.is_slide_no_head_and_delay else wait
            if len(segments) == 1:
                first = segments[0]
                slide_info = SlideInfo(
                    pattern=first.pattern,
                    start_position=key_id,
                    end_position=first.end_position,
                    duration=wait + total,
                    delay=delay,
                )
                duration = wait + total
            else:
                share = total / len(segments)
                result_notes: list[MaimaiNote] = []
                for segment_index, segment in enumerate(segments):
                    is_first = segment_index == 0
                    segment_duration = wait + share if is_first else share
                    start_position = (
                        key_id if is_first else segments[segment_index - 1].end_position
                    )
                    result_notes.append(
                        MaimaiNote(
                            source_index=self._source_index,
                            start_time=timing,
                            duration=segment_duration,
                            type=MaimaiNoteType.SLIDE,
                            key_id=start_position,
                            is_break=flags.is_break,
                            is_fireworks=flags.is_hanabi,
                            is_ex=flags.is_ex,
                            slide_info=SlideInfo(
                                pattern=segment.pattern,
                                start_position=start_position,
                                end_position=segment.end_position,
                                duration=segment_duration,
                                delay=0.0 if not is_first else delay,
                            ),
                        )
                    )
                    self._source_index += 1
                return result_notes

        note = MaimaiNote(
            source_index=self._source_index,
            start_time=timing,
            duration=duration,
            type=note_type,
            key_id=key_id,
            key_group=key_group,
            is_break=flags.is_break,
            is_fireworks=flags.is_hanabi,
            is_ex=flags.is_ex,
            slide_info=slide_info,
        )
        self._source_index += 1
        return [note]

    @staticmethod
    def _detect_flags(token: str) -> tuple[str, _NoteFlags]:
        """Port MajSimai ``NoteFlag.Detect`` including b/m position rules."""
        flags = _NoteFlags()
        cleaned: list[str] = []

        for _index, char in enumerate(token):
            if char in SLIDE_CHARACTERS:
                flags.is_slide = True
            elif char in TOUCH_AREAS:
                flags.is_touch = True
            elif char == "f":
                flags.is_hanabi = True
            elif char == "x":
                flags.is_ex = True
                continue
            elif char == "h":
                flags.is_hold = True
            elif char == "!":
                flags.is_slide_no_head = True
                continue
            elif char == "?":
                flags.is_slide_no_head_and_delay = True
                continue
            elif char == "$":
                continue
            elif char == "b":
                flags.is_break = True
                continue
            elif char == "m":
                if not flags.is_slide:
                    flags.is_mine = True
                continue
            elif char == "c":
                continue
            cleaned.append(char)

        return "".join(cleaned), flags

    def _hold_time(self, bpm: float, note_text: str) -> float:
        match = DURATION_RE.search(note_text)
        if match is None:
            return 0.0
        body = match.group(1)
        parts = body.split("#")
        if len(parts) == 1:
            return self._ratio_seconds(bpm, body)
        if len(parts) == 2:
            custom_bpm_text, duration_text = parts
            if not custom_bpm_text:
                return self._number(duration_text)
            if not duration_text:
                raise ValueError("hold duration is empty")
            return self._ratio_seconds(self._number(custom_bpm_text), duration_text)
        raise ValueError(f"unsupported hold duration: [{body}]")

    def _slide_times(self, bpm: float, note_text: str) -> tuple[float, float]:
        bodies = DURATION_RE.findall(note_text)
        if not bodies:
            raise ValueError("slide duration is absent")

        total = 0.0
        custom_wait_bpm: float | None = None
        for body in bodies:
            parts = body.split("#")
            if len(parts) == 1:
                total += self._ratio_seconds(bpm, body)
            elif len(parts) == 2:
                custom_bpm_text, duration_text = parts
                if not custom_bpm_text or not duration_text:
                    raise ValueError(f"invalid slide duration: [{body}]")
                custom_bpm = self._number(custom_bpm_text)
                direct = self._try_number(duration_text)
                total += (
                    direct if direct is not None else self._ratio_seconds(custom_bpm, duration_text)
                )
                if custom_wait_bpm is None:
                    custom_wait_bpm = custom_bpm
            elif len(parts) == 3:
                wait_text, middle, duration_text = parts
                if not wait_text or middle or not duration_text:
                    raise ValueError(f"invalid slide duration: [{body}]")
                explicit_wait = self._number(wait_text)
                direct = self._try_number(duration_text)
                total += direct if direct is not None else self._ratio_seconds(bpm, duration_text)
                if custom_wait_bpm is None:
                    custom_wait_bpm = math.inf if explicit_wait == 0 else 60.0 / explicit_wait
            elif len(parts) == 4:
                wait_text, empty, custom_bpm_text, ratio = parts
                if not wait_text or empty or not custom_bpm_text or ratio:
                    raise ValueError(f"invalid slide duration: [{body}]")
                explicit_wait = self._number(wait_text)
                total += self._ratio_seconds(self._number(custom_bpm_text), ratio)
                if custom_wait_bpm is None:
                    custom_wait_bpm = math.inf if explicit_wait == 0 else 60.0 / explicit_wait
            else:
                raise ValueError(f"unsupported slide duration: [{body}]")

        wait_bpm = custom_wait_bpm if custom_wait_bpm is not None else bpm
        if wait_bpm == math.inf:
            wait = 0.0
        elif wait_bpm == 0:
            raise ValueError("slide wait requires a non-zero BPM")
        else:
            wait = 60.0 / wait_bpm
        return wait, total

    @staticmethod
    def _v_fold_direction(start: int, fold: int) -> str:
        """Classify a V-fold as left (incoming) or right (reflected).

        Follows the MA2 convention where the incoming fold sits two keys
        counter-clockwise from the start key and the reflected fold two keys
        clockwise.
        """
        delta = (fold - start) % 8
        if delta in {1, 2, 3}:
            return "v_fold_right"
        if delta in {5, 6, 7}:
            return "v_fold_left"
        return "v_fold_left" if delta == 4 else "v_fold_right"

    def _extract_slide_segments(
        self,
        note_text: str,
        start_position: int,
    ) -> list[SlideInfo]:
        """Extract collapsed per-segment geometry from a slide body."""
        cursor = 1
        current = start_position
        segments: list[SlideInfo] = []

        while cursor < len(note_text):
            char = note_text[cursor]
            if char == "[":
                end = note_text.find("]", cursor)
                if end == -1:
                    raise ValueError("unclosed slide duration")
                cursor = end + 1
                continue
            if char in {"h", "f"}:
                cursor += 1
                continue

            shape = next(
                (mark for mark in SLIDE_MARKS if note_text.startswith(mark, cursor)),
                None,
            )
            if shape is None:
                cursor += 1
                continue
            cursor += len(shape)

            pattern: str
            if shape == "V":
                if cursor >= len(note_text) or note_text[cursor] not in "12345678":
                    raise ValueError("V slide is missing its inflection position")
                fold = int(note_text[cursor])
                cursor += 1
                pattern = SimaiParser._v_fold_direction(current, fold)
            else:
                pattern = SLIDE_PATTERN_MAP[shape]
            if cursor >= len(note_text) or note_text[cursor] not in "12345678":
                raise ValueError(f"slide {shape!r} is missing its end position")
            end_position = int(note_text[cursor])
            cursor += 1

            duration: float | None = None
            segments.append(
                SlideInfo(
                    pattern=pattern,
                    start_position=current,
                    end_position=end_position,
                    duration=duration,
                    delay=0.0,
                )
            )
            current = end_position
        return segments

    @staticmethod
    def _ratio_seconds(bpm: float, ratio: str) -> float:
        if bpm == 0 or not math.isfinite(bpm):
            raise ValueError("duration BPM must be finite and non-zero")
        parts = ratio.split(":")
        if len(parts) != 2:
            raise ValueError(f"duration ratio must be x:y, got {ratio!r}")
        division = int(parts[0])
        count = int(parts[1])
        if division == 0:
            raise ValueError("duration division must not be zero")
        return (60.0 / bpm) * (4.0 / division) * count

    @staticmethod
    def _number(value: str) -> float:
        number = float(value.strip())
        if not math.isfinite(number):
            raise ValueError("number must be finite")
        return number

    @classmethod
    def _try_number(cls, value: str) -> float | None:
        try:
            return cls._number(value)
        except ValueError:
            return None

    def _warn(self, chart: MaimaiChart, state: _ParserState, message: str) -> None:
        if len(chart.warnings) < self.max_warnings:
            chart.warnings.append(f"cell {state.cell_index}: {message}")
