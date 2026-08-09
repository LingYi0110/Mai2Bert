from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from loguru import logger

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[module]}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
)
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {process.id}:{thread.id} | "
    "{extra[module]}:{line} | {message}"
)

_configured = False


class LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.bind(module=record.name).opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


def setup_logger(
    level: str = "INFO",
    file: str | Path | None = None,
    *,
    log_dir: str | Path | None = None,
    experiment: str | None = None,
    rotation: str = "20 MB",
    retention: str = "14 days",
    force: bool = False,
) -> None:
    global _configured
    if _configured and not force:
        return

    logger.remove()
    logger.configure(extra={"module": "mai2bert"})
    logger.add(
        sys.stderr,
        level=os.getenv("MAI2BERT_LOG_LEVEL", level),
        format=CONSOLE_FORMAT,
        colorize=True,
    )
    output_file = os.getenv("MAI2BERT_LOG_FILE") or file
    if output_file is None and log_dir is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = Path(log_dir) / f"{experiment or 'mai2bert'}-{timestamp}.log"
    if output_file is not None:
        destination = Path(output_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            destination,
            level=os.getenv("MAI2BERT_LOG_LEVEL", level),
            format=FILE_FORMAT,
            rotation=rotation,
            retention=retention,
            compression="zip",
            enqueue=True,
        )

    logging.basicConfig(handlers=[LoguruHandler()], level=0, force=True)
    for name in ("lightning", "torch", "h5py"):
        logging.getLogger(name).handlers = [LoguruHandler()]
        logging.getLogger(name).propagate = False
    _configured = True


def get_logger(name: str | None = None) -> Any:
    return logger.bind(module=name or "mai2bert")
