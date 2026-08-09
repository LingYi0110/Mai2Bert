#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Run against src/ directly; keeps preprocessing resolving to the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lib.config import load_app_config
from lib.utils.logging import get_logger, setup_logger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preprocessing",
        description="Prepare the processed binary store from raw chart JSON corpora.",
    )
    parser.add_argument("--config", required=True, help="path to the prepare stage YAML config")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-list override; may be repeated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_app_config(args.config, args.set)
    setup_logger(
        config.logging.level,
        config.logging.file,
        log_dir=config.logging.log_dir,
        experiment=config.experiment,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
    )
    logger = get_logger("preprocessing")
    logger.info("Starting prepare stage")
    from preprocessing.prepare import run_prepare

    result = run_prepare(config)
    logger.info("Completed prepare stage -> {}", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
