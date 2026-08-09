from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from lib.parser import Ma2Parser, SimaiParser
from lib.parser.model import MaimaiChart


def _collect_sources(paths: list[str]) -> list[Path]:
    sources: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            sources.append(path)
        elif path.is_dir():
            sources.extend(sorted(path.rglob("*.ma2")))
            sources.extend(sorted(path.rglob("*.txt")))
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(set(sources))


def _write_chart(
    chart: MaimaiChart,
    output_name: str,
    output_dir: Path,
    *,
    overwrite: bool,
) -> bool:
    target = output_dir / output_name
    if target.exists() and not overwrite:
        return False
    target.write_text(
        json.dumps(chart.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def _write_index(output_dir: Path, converted: list[str], skipped: list[str]) -> None:
    index_path = output_dir / "index.csv"
    entries: dict[str, list[str]] = {}
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                filename = (row.get("file") or "").strip()
                if not filename:
                    continue
                entries[filename] = [
                    filename,
                    (row.get("label_type") or "").strip(),
                    (row.get("label") or "").strip(),
                ]
    for filename in converted + skipped:
        entries.setdefault(filename, [filename, "", ""])
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "label_type", "label"))
        for filename in sorted(entries):
            writer.writerow(entries[filename])


def convert_ma2(
    source: Path,
    output_dir: Path,
    *,
    strict: bool,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    chart = Ma2Parser.from_file(source, strict=strict)
    output_name = f"{source.stem}.json"
    if _write_chart(chart, output_name, output_dir, overwrite=overwrite):
        return [output_name], []
    return [], [output_name]


def convert_simai(
    source: Path,
    output_dir: Path,
    *,
    difficulty: int | None,
    strict: bool,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    parsed = SimaiParser.from_file(source, difficulty=difficulty, strict=strict)
    if difficulty is not None:
        charts: dict[int, MaimaiChart] = {difficulty: parsed}
    else:
        charts = parsed.charts
    written: list[str] = []
    skipped: list[str] = []
    for index, chart in sorted(charts.items()):
        output_name = f"{source.stem}_{index}.json"
        if _write_chart(chart, output_name, output_dir, overwrite=overwrite):
            written.append(output_name)
        else:
            skipped.append(output_name)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="convert_charts",
        description="Convert MA2/Simai chart files to the simplified chart JSON layout.",
    )
    parser.add_argument("sources", nargs="+", help="chart files or directories to convert")
    parser.add_argument(
        "-o",
        "--output",
        default="json",
        help="output directory (default: json/)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 8),
        help="for Simai files, export only this difficulty (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort on the first parse error instead of collecting failures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing output files instead of skipping them",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _collect_sources(args.sources)
    if not sources:
        print(f"no .ma2 or .txt files found under {args.sources}", file=sys.stderr)
        return 1

    converted: list[str] = []
    skipped_names: list[str] = []
    failures: list[tuple[Path, str]] = []
    for source in sources:
        try:
            if source.suffix.lower() == ".ma2":
                written, skipped = convert_ma2(
                    source,
                    output_dir,
                    strict=args.strict,
                    overwrite=args.overwrite,
                )
            else:
                written, skipped = convert_simai(
                    source,
                    output_dir,
                    difficulty=args.difficulty,
                    strict=args.strict,
                    overwrite=args.overwrite,
                )
        except Exception as error:
            failures.append((source, f"{type(error).__name__}: {error}"))
            continue
        converted.extend(written)
        skipped_names.extend(skipped)

    _write_index(output_dir, converted, skipped_names)
    print(f"converted {len(converted)} chart(s) to {output_dir}")
    if skipped_names:
        print(f"skipped {len(skipped_names)} existing file(s) (pass --overwrite to replace)")
    for source, message in failures:
        print(f"failed {source}: {message}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} file(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
