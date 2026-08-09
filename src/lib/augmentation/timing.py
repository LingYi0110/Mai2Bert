from __future__ import annotations

import math
import random
from dataclasses import replace

from lib.parser.model import MaimaiChart, MaimaiNote


def _jittered_times(
    timestamps: list[float],
    max_jitter_seconds: float,
    rng: random.Random,
) -> list[float]:
    """Generate order-preserving jittered times for sorted unique timestamps."""
    if not timestamps:
        return []

    result: list[float] = []
    for index, timestamp in enumerate(timestamps):
        lower = max(0.0, timestamp - max_jitter_seconds)
        upper = timestamp + max_jitter_seconds
        if index + 1 < len(timestamps):
            # Groups stay below the next original midpoint so order is preserved.
            upper = min(upper, (timestamp + timestamps[index + 1]) / 2.0)
        if result:
            epsilon = math.ulp(max(abs(timestamp), abs(result[-1]), 1.0))
            lower = max(lower, result[-1] + epsilon)
        # Original timestamp is always feasible; fallback guards round-off.
        if lower > upper:
            lower = upper = timestamp
        result.append(rng.uniform(lower, upper))
    return result


def jitter_note_timing(
    note: MaimaiNote,
    delta_seconds: float,
) -> MaimaiNote:
    """Return a note shifted in time without changing its durations."""
    return replace(note, start_time=note.start_time + delta_seconds)


def jitter_chart_timing(
    chart: MaimaiChart,
    max_jitter_seconds: float,
    *,
    seed: int | None = None,
) -> MaimaiChart:
    """Return a chart with small, order-preserving onset-time noise.

    Notes sharing an original timestamp receive the same offset.  For each
    distinct timestamp, the requested jitter interval is clipped to the
    midpoint with its neighbours.  Consequently, timestamps remain strictly
    ordered when they were strictly ordered, while simultaneous notes remain
    simultaneous.  The input chart is never mutated.
    """
    if max_jitter_seconds < 0.0:
        raise ValueError("max_jitter_seconds must be nonnegative")

    notes = sorted(chart.notes, key=lambda note: (note.start_time, note.source_index))
    groups: list[list[MaimaiNote]] = []
    for note in notes:
        if not groups or note.start_time != groups[-1][0].start_time:
            groups.append([note])
        else:
            groups[-1].append(note)

    timestamps = [group[0].start_time for group in groups]
    rng = random.Random(seed)
    jittered = _jittered_times(timestamps, max_jitter_seconds, rng)
    shifts = {
        timestamp: new_timestamp - timestamp
        for timestamp, new_timestamp in zip(timestamps, jittered, strict=True)
    }

    jittered_notes = [jitter_note_timing(note, shifts[note.start_time]) for note in chart.notes]
    return replace(
        chart,
        notes=jittered_notes,
        warnings=list(chart.warnings),
    )


__all__ = ["jitter_chart_timing", "jitter_note_timing"]
