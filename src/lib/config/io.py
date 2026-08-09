from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


def _load_with_bases(path: Path, stack: tuple[Path, ...]) -> DictConfig:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"circular config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"config file does not exist: {path}")

    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"config root must be a mapping: {path}")

    base_value = loaded.get("bases", [])
    if isinstance(base_value, str):
        bases = [base_value]
    elif isinstance(base_value, (list, tuple, ListConfig)):
        bases = list(base_value)
    else:
        raise TypeError(f"bases must be a string or list in {path}")

    merged = OmegaConf.create({})
    for base in bases:
        if not isinstance(base, str):
            raise TypeError(f"base paths must be strings in {path}")
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = cast(
            DictConfig,
            OmegaConf.merge(merged, _load_with_bases(base_path, (*stack, path))),
        )

    own = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(loaded, resolve=False)))
    if "bases" in own:
        del own["bases"]
    return cast(DictConfig, OmegaConf.merge(merged, own))


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    config = _load_with_bases(Path(path), ())
    if overrides:
        config = cast(
            DictConfig,
            OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides))),
        )
    OmegaConf.resolve(config)
    return config


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved config root must be a mapping")
    return cast(dict[str, Any], value)


def save_config(config: DictConfig | dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    OmegaConf.save(value, destination, resolve=True)
