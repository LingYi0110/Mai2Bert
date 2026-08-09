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
        prog="train", description="Pretrain or fine-tune the chart model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("pretrain", "finetune"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="OmegaConf dot-list override; may be repeated",
        )
        child.add_argument(
            "--resume",
            metavar="CHECKPOINT",
            help=(
                'resume the complete training state; pass a .ckpt path, "last", '
                "or an experiment directory (newest last-<epoch>.ckpt is used)"
            ),
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
    logger = get_logger("train")
    logger.info("Starting {} stage", args.command)
    if args.command == "pretrain":
        from training.pretrain import run_pretrain

        run_pretrain(config, resume_checkpoint=args.resume)
    else:
        from training.regression import run_finetune

        run_finetune(config, resume_checkpoint=args.resume)
    logger.info("Completed {} stage", args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
