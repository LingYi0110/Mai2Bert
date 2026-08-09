from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

from lib.parser import ChartFormat, Ma2Parser, SimaiParser

from .predict import PredictionResult, _chart_item


def _scan_inputs(
    raw_inputs: list[str],
    *,
    format: ChartFormat,
    limit: int | None = None,
) -> list[Path]:
    pattern = "*.ma2" if format == "ma2" else "*.txt"
    found: list[Path] = []
    for raw in raw_inputs:
        if any(character in raw for character in "*?["):
            found.extend(Path(match) for match in glob.glob(raw, recursive=True))
            continue
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"input does not exist: {raw}")
    deduplicated = list(dict.fromkeys(found))
    if limit is not None:
        deduplicated = deduplicated[:limit]
    return deduplicated


def _build_items(
    files: list[Path],
    *,
    format: ChartFormat,
    difficulty: int | None,
    config: Any,
) -> tuple[list[dict[str, Any]], list[PredictionResult]]:
    items: list[dict[str, Any]] = []
    records: list[PredictionResult] = []
    if format == "ma2":
        for file in files:
            chart = Ma2Parser.from_file(file)
            items.append(_chart_item(chart, config))
            records.append(
                PredictionResult(
                    source=str(file),
                    format="ma2",
                    difficulty=None,
                    predicted=float("nan"),
                    sigma=float("nan"),
                    n_notes=len(chart.notes),
                    n_windows=0,
                )
            )
        return items, records
    for file in files:
        parsed = SimaiParser.from_file(file)
        requested = [difficulty] if difficulty is not None else sorted(parsed.charts)
        for chart_difficulty in requested:
            if chart_difficulty not in parsed.charts:
                print(f"  [skip] {file}: difficulty {chart_difficulty} absent", file=sys.stderr)
                continue
            chart = parsed.get_chart(chart_difficulty)
            items.append(_chart_item(chart, config))
            records.append(
                PredictionResult(
                    source=str(file),
                    format="simai",
                    difficulty=chart_difficulty,
                    predicted=float("nan"),
                    sigma=float("nan"),
                    n_notes=len(chart.notes),
                    n_windows=0,
                )
            )
    return items, records


def _write_outputs(
    output: Path,
    results: list[PredictionResult],
    *,
    format: ChartFormat,
    checkpoint: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("source,format,difficulty,n_notes,predicted,sigma\n")
        for result in results:
            stream.write(
                f"{result.source},{result.format},{result.difficulty or ''},"
                f"{result.n_notes},{result.predicted:.4f},{result.sigma:.4f}\n"
            )
    payload = {
        "checkpoint": checkpoint,
        "format": format,
        "count": len(results),
        "results": [
            {
                "source": result.source,
                "format": result.format,
                "difficulty": result.difficulty,
                "n_notes": result.n_notes,
                "predicted": result.predicted,
                "sigma": result.sigma,
            }
            for result in results
        ],
    }
    (output.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
