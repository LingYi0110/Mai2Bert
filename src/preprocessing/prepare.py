from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lib.augmentation import jitter_chart_timing, rotate_chart
from lib.config.io import save_config
from lib.config.schema import AppConfig
from lib.parser import MaimaiChart, MaimaiNoteType
from lib.utils.logging import get_logger
from preprocessing.discovery import (
    ChartKey,
    ChartRecord,
    discover_json_corpus,
    load_json_chart,
)
from preprocessing.representation import (
    EventArrays,
    chart_to_arrays,
    representation_schema,
    schema_hash,
)
from preprocessing.split import SplitAssignment, split_supervised_groups
from preprocessing.store import STORE_FORMAT_VERSION, ProcessedStoreWriter

logger = get_logger("preprocessing.prepare")

PREPROCESSING_IMPLEMENTATION_VERSION = 2
PARSER_IMPLEMENTATION_VERSION = 2


@dataclass(frozen=True, slots=True)
class PreparedVariant:
    rotation: str
    arrays: EventArrays | None = None
    physical_notes: int = 0
    slide_segments: int = 0
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedChart:
    parser_warnings: tuple[str, ...] = ()
    variants: tuple[PreparedVariant, ...] = ()


def _stable_augmentation_seed(split_seed: int, chart_key: ChartKey, rotation: str) -> int:
    """Return a process-independent seed for one training chart variant."""
    digest = hashlib.sha256(f"timing-jitter:{split_seed}:{chart_key}:{rotation}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _hash_file(path: Path, digest: Any) -> None:
    """Hash one source path and its contents into ``digest``."""
    resolved = path.resolve()
    digest.update(resolved.as_posix().encode("utf-8"))
    digest.update(b"\0")
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    digest.update(b"\0")


def _fingerprint(config: AppConfig, charts: list[ChartRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(f"preprocessing:{PREPROCESSING_IMPLEMENTATION_VERSION}".encode("ascii"))
    digest.update(schema_hash().encode("ascii"))
    digest.update(
        json.dumps(
            {
                "datasets": config.datasets,
                "split": config.split.model_dump(mode="json"),
                "representation": config.representation.model_dump(mode="json"),
                "augmentation": config.augmentation.model_dump(mode="json"),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    # Hash each indexed JSON source exactly once.
    source_paths = {
        (config.paths.raw_data / dataset / "index.csv").resolve() for dataset in config.datasets
    }
    source_paths.update(chart.path.resolve() for chart in charts)
    for path in sorted(source_paths, key=lambda item: item.as_posix()):
        _hash_file(path, digest)
    return digest.hexdigest()[:16]


def _rotations_for_chart(
    chart: ChartRecord,
    split: str | None,
    pretraining: bool,
    config: AppConfig,
) -> list[str]:
    rotations: list[str] = []
    if pretraining or split == "train":
        rotations.extend(config.augmentation.train_rotations)
    if split in {"validation", "test"}:
        rotations.extend(config.augmentation.evaluation_rotations)
    return list(dict.fromkeys(rotations))


def _parse_chart(record: ChartRecord) -> tuple[MaimaiChart, list[str]]:
    parsed = load_json_chart(record.path)
    return parsed, list(parsed.warnings)


def _variant_counts(chart: MaimaiChart, arrays: EventArrays) -> tuple[int, int]:
    # note_start is the representation's authoritative physical-note boundary.
    physical_notes = int(arrays.note_start.sum())
    slides = sum(1 for note in chart.notes if note.type is MaimaiNoteType.SLIDE)
    return physical_notes, slides


def _prepare_chart(
    chart: ChartRecord,
    rotations: tuple[str, ...],
    *,
    split_seed: int,
    jitter_seconds: float,
    continuous_clip: float,
) -> PreparedChart:
    """Parse and build all variants for one JSON chart outside the HDF5 writer."""
    try:
        parsed, parser_warnings = _parse_chart(chart)
    except Exception as error:
        return PreparedChart(
            variants=tuple(
                PreparedVariant(rotation=rotation, reason="parser_error", error=str(error))
                for rotation in rotations
            ),
        )

    prepared: list[PreparedVariant] = []
    for rotation in rotations:
        try:
            rotated = rotate_chart(parsed, rotation)
            jittered = jitter_chart_timing(
                rotated,
                jitter_seconds,
                seed=_stable_augmentation_seed(split_seed, chart.key, rotation),
            )
            arrays = chart_to_arrays(jittered, clip=continuous_clip)
            if arrays.length == 0:
                prepared.append(
                    PreparedVariant(
                        rotation=rotation,
                        reason="empty_chart",
                        error="representation contains zero events",
                    )
                )
                continue
            physical_notes, slide_segments = _variant_counts(jittered, arrays)
            prepared.append(
                PreparedVariant(
                    rotation=rotation,
                    arrays=arrays,
                    physical_notes=physical_notes,
                    slide_segments=slide_segments,
                )
            )
        except Exception as error:
            prepared.append(
                PreparedVariant(rotation=rotation, reason="representation_error", error=str(error))
            )
    return PreparedChart(parser_warnings=tuple(parser_warnings), variants=tuple(prepared))


def _prepare_chart_worker(
    args: tuple[ChartRecord, tuple[str, ...], int, float, float],
) -> PreparedChart:
    """Process-pool entry point; keep it module-level for Windows spawn."""
    chart, rotations, split_seed, jitter_seconds, continuous_clip = args
    return _prepare_chart(
        chart,
        rotations,
        split_seed=split_seed,
        jitter_seconds=jitter_seconds,
        continuous_clip=continuous_clip,
    )


def _skip_record(
    chart: ChartRecord,
    rotation: str,
    reason: str,
    error: BaseException | str,
) -> dict[str, Any]:
    return {
        "dataset": chart.dataset,
        "music_id": chart.music_id,
        "path": str(chart.path),
        "rotation": rotation,
        "reason": reason,
        "error": str(error),
    }


def _split_assignment_hash(
    assignment: SplitAssignment,
    charts: list[ChartRecord],
    seed: int,
) -> str:
    """Content hash of the split assignment, recorded in the provenance report.

    The assignment itself lives with the processed store (each variant row
    carries its ``split``/``pretraining`` flags in ``variants.jsonl``), so no
    separate manifest file is written.
    """
    rows = [
        {
            "dataset": chart.dataset,
            "music_id": chart.music_id,
            "supervised_split": assignment.supervised_keys.get(chart.key),
            "pretraining": chart.key in assignment.pretraining_keys,
        }
        for chart in sorted(charts, key=lambda item: item.key)
    ]
    payload = {
        "version": 2,
        "seed": seed,
        "group_splits": dict(sorted(assignment.group_splits.items())),
        "charts": rows,
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
    return hashlib.sha256(serialized.encode()).hexdigest()


def run_prepare(config: AppConfig) -> Path:
    charts, discovery_report = discover_json_corpus(config.paths.raw_data, config.datasets)
    logger.info(
        "Discovered {} JSON charts ({} labeled, {} unlabeled)",
        discovery_report.standard_charts,
        discovery_report.labeled_charts,
        discovery_report.unlabeled_charts,
    )
    assignment = split_supervised_groups(
        charts,
        ratios=(
            config.split.train_ratio,
            config.split.validation_ratio,
            config.split.test_ratio,
        ),
        seed=config.split.seed,
        threshold=config.split.high_diff_threshold,
        bin_width=config.split.bin_width,
        rare_bin_max_groups=config.split.rare_bin_max_groups,
    )
    split_hash = _split_assignment_hash(assignment, charts, config.split.seed)
    fingerprint = _fingerprint(config, charts)
    output = config.dataset_binary_dir
    if output.exists():
        raise FileExistsError(f"immutable processed dataset already exists: {output}")
    output.mkdir(parents=True)

    variants_path = output / "variants.jsonl"
    written = 0
    skipped: list[dict[str, Any]] = []
    parser_warning_records: list[dict[str, Any]] = []
    tasks: list[tuple[ChartRecord, tuple[str, ...], int, float, float]] = []
    for chart in charts:
        split = assignment.supervised_keys.get(chart.key)
        pretraining = chart.key in assignment.pretraining_keys
        rotations = tuple(_rotations_for_chart(chart, split, pretraining, config))
        jitter_seconds = (
            config.augmentation.train_timing_jitter_seconds
            if split == "train" or pretraining
            else 0.0
        )
        tasks.append(
            (
                chart,
                rotations,
                config.split.seed,
                jitter_seconds,
                config.representation.continuous_clip,
            )
        )

    progress = tqdm(
        total=len(charts),
        unit="chart",
        desc="prepare",
        dynamic_ncols=True,
        file=sys.stderr,
    )
    executor = (
        ProcessPoolExecutor(max_workers=config.data.preprocessing_workers)
        if config.data.preprocessing_workers > 0
        else None
    )
    try:
        results = (
            executor.map(_prepare_chart_worker, tasks, chunksize=4)
            if executor is not None
            else (
                _prepare_chart(
                    chart,
                    rotations,
                    split_seed=split_seed,
                    jitter_seconds=jitter_seconds,
                    continuous_clip=continuous_clip,
                )
                for chart, rotations, split_seed, jitter_seconds, continuous_clip in tasks
            )
        )
        with (
            ProcessedStoreWriter(output / "events.h5") as writer,
            variants_path.open("w", encoding="utf-8", newline="\n") as manifest,
        ):
            for chart, task, prepared in zip(charts, tasks, results, strict=True):
                _, rotations, _, _, _ = task
                split = assignment.supervised_keys.get(chart.key)
                pretraining = chart.key in assignment.pretraining_keys
                parser_warnings = list(prepared.parser_warnings)
                if parser_warnings:
                    parser_warning_records.append(
                        {
                            "dataset": chart.dataset,
                            "music_id": chart.music_id,
                            "path": str(chart.path),
                            "warnings": parser_warnings,
                        }
                    )
                for variant in prepared.variants:
                    if variant.arrays is None:
                        reason = variant.reason or "worker_error"
                        error = variant.error or "worker returned no arrays"
                        skipped.append(_skip_record(chart, variant.rotation, reason, error))
                        progress.write(
                            f"[skip:{reason}] {chart.key} rot={variant.rotation}: {error}"[:200]
                        )
                        continue
                    row = writer.append(variant.arrays)
                    record = {
                        "row": row,
                        "dataset": chart.dataset,
                        "music_id": chart.music_id,
                        "rotation": variant.rotation,
                        "split": split,
                        "pretraining": pretraining,
                        "difficulty_const": chart.difficulty_const,
                        "length": variant.arrays.length,
                        "label_type": chart.label_type,
                        "format": chart.format,
                        "physical_notes": variant.physical_notes,
                        "slide_segments": variant.slide_segments,
                        "parser_warnings": len(parser_warnings),
                    }
                    manifest.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                    written += 1
                progress.update(1)
    except Exception as error:
        # Keep worker failures diagnosable instead of silent.
        logger.exception("Preprocessing worker failed: {}", error)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        progress.close()

    skip_reasons = Counter(item["reason"] for item in skipped)
    if skipped:
        logger.warning("Skipped {} chart variants: {}", len(skipped), dict(skip_reasons))

    (output / "schema.json").write_text(
        json.dumps(representation_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "fingerprint": fingerprint,
        "schema_hash": schema_hash(),
        "split_hash": split_hash,
        "source_charts": len(charts),
        "datasets": config.datasets,
        "variants": written,
        "skipped_variants": len(skipped),
        "parser_version": PARSER_IMPLEMENTATION_VERSION,
        "preprocessing_version": PREPROCESSING_IMPLEMENTATION_VERSION,
        "store_version": STORE_FORMAT_VERSION,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        **asdict(discovery_report),
        "canonical_groups": len({chart.music_id for chart in charts}),
        "processed_variants": written,
        "skipped_variants": len(skipped),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "skips": skipped,
        "parser_warnings": sum(len(record["warnings"]) for record in parser_warning_records),
        "parser_warning_records": parser_warning_records,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_config(config.model_dump(mode="json"), output / "resolved.yaml")
    logger.info("Prepared dataset for experiment {} at {}", config.experiment, output)
    return output
