from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from lib.config.io import config_to_dict, load_config
from lib.config.schema import AppConfig


def load_app_config(path: str | Path, overrides: Sequence[str] = ()) -> AppConfig:
    raw = load_config(path, overrides)
    return AppConfig.model_validate(config_to_dict(raw))
