#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Root scripts run against the src/ layout directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lib.config import load_app_config
from lib.utils.logging import get_logger, setup_logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate", description="Evaluate the finetuned regressor on the held-out test split."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to evaluate YAML config; else use checkpoint's sidecar config.yaml",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-list override; may be repeated",
    )
    parser.add_argument("--checkpoint", default=None, help="finetuned regression checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = list(args.set)
    if args.checkpoint:
        overrides.append(f"paths.checkpoint={args.checkpoint}")
    if args.config:
        config = load_app_config(args.config, overrides)
    else:
        from inference.loader import load_inference_config

        config = load_inference_config(args.checkpoint, overrides=overrides)
    setup_logger(
        config.logging.level,
        config.logging.file,
        log_dir=config.logging.log_dir,
        experiment=config.experiment,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
    )
    logger = get_logger("evaluate")
    logger.info("Starting evaluate stage")
    from training.regression import run_evaluate

    metrics = run_evaluate(config)
    print(metrics)
    logger.info("Completed evaluate stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
