#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Root scripts run against the src/ layout directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lib.config import AppConfig, load_app_config
from lib.utils.logging import get_logger, setup_logger


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="path to YAML config; else use the checkpoint's sidecar config.yaml",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-list override; may be repeated",
    )
    parser.add_argument("--checkpoint", default=None, help="finetuned regression checkpoint")


def _run_predict(args: argparse.Namespace, config: AppConfig) -> None:
    from inference.loader import load_module
    from inference.predict import predict_file

    module = load_module(config, args.checkpoint)
    result = predict_file(
        module,
        config,
        args.chart,
        format=args.format,
        difficulty=args.difficulty,
    )
    print(f"{result.predicted:.6f}")


def _run_batch(args: argparse.Namespace, config: AppConfig) -> None:
    from dataclasses import replace

    from inference.batch import _build_items, _scan_inputs, _write_outputs
    from inference.loader import load_module
    from inference.predict import predict_items
    from lib.parser import ChartFormat

    resolved_format: ChartFormat = args.format or "simai"
    files = _scan_inputs(args.input, format=resolved_format, limit=args.limit)
    if not files:
        print("no chart files matched the inputs", file=sys.stderr)
        raise SystemExit(2)
    print(f"found {len(files)} files ({resolved_format})")

    device = None if args.device == "auto" else args.device
    module = load_module(config, args.checkpoint, device=device)
    items, records = _build_items(
        files, format=resolved_format, difficulty=args.difficulty, config=config
    )
    if not items:
        print("no parseable charts remain after expansion", file=sys.stderr)
        raise SystemExit(2)

    predictions = predict_items(module, items, config, batch_size=args.batch_size)
    results = [
        replace(record, predicted=mean, sigma=sigma)
        for record, (mean, sigma) in zip(records, predictions, strict=True)
    ]
    _write_outputs(
        args.output,
        results,
        format=resolved_format,
        checkpoint=str(args.checkpoint or config.transfer.checkpoint or ""),
    )
    print(f"predicted {len(results)} charts -> {args.output} (+ .json)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer", description="Predict chart difficulty from MA2 or Simai chart files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="predict one chart file")
    _common_arguments(predict)
    predict.add_argument("--chart", required=True)
    predict.add_argument("--format", choices=("ma2", "simai"))
    predict.add_argument("--difficulty", type=int, choices=range(1, 8))

    batch = subparsers.add_parser("batch", help="predict many chart files")
    _common_arguments(batch)
    batch.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="PATH",
        help="chart file, directory (recursive), glob, or comma/space separated list",
    )
    batch.add_argument(
        "--format",
        choices=("ma2", "simai"),
        default=None,
        help="defaults to simai for .txt files, ma2 otherwise",
    )
    batch.add_argument("--difficulty", type=int, default=None, help="simai difficulty 1..7")
    batch.add_argument("--output", type=Path, default=Path("predictions.csv"))
    batch.add_argument("--batch-size", type=int, default=32)
    batch.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    batch.add_argument("--limit", type=int, default=None, help="max files (testing)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.config:
        config = load_app_config(args.config, args.set)
    else:
        from inference.loader import load_inference_config

        config = load_inference_config(args.checkpoint, overrides=args.set)
    setup_logger(
        config.logging.level,
        config.logging.file,
        log_dir=config.logging.log_dir,
        experiment=config.experiment,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
    )
    logger = get_logger("infer")
    logger.info("Starting predict stage ({})", args.command)
    if args.command == "predict":
        _run_predict(args, config)
    else:
        _run_batch(args, config)
    logger.info("Completed predict stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
