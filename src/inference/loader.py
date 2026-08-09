from __future__ import annotations

from pathlib import Path

import torch

from lib.config import AppConfig, load_app_config
from preprocessing.representation import schema_hash
from training.common import architecture_hash
from training.regression import RegressionModule


def resolve_checkpoint(
    config: AppConfig,
    checkpoint: str | Path | None = None,
) -> Path:
    resolved = checkpoint or config.paths.checkpoint or config.transfer.checkpoint
    if resolved is None:
        raise ValueError("inference requires a checkpoint (--checkpoint or paths.checkpoint)")
    return Path(resolved)


def load_inference_config(
    checkpoint: str | Path | None,
    *,
    overrides: list[str] | None = None,
) -> AppConfig:
    if checkpoint is None:
        raise ValueError("inference requires --config or a checkpoint that carries a config.yaml")
    sidecar = Path(checkpoint).parent / "config.yaml"
    if not sidecar.is_file():
        raise ValueError(
            f"no sidecar config found next to checkpoint: {sidecar}; "
            "pass --config explicitly or train with a config.yaml archive"
        )
    return load_app_config(sidecar, overrides or ())


def load_module(
    config: AppConfig,
    checkpoint: str | Path | None = None,
    *,
    device: str | None = None,
) -> RegressionModule:
    resolved = resolve_checkpoint(config, checkpoint)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    try:
        hyperparameters = payload.get("hyper_parameters", {})
        if hyperparameters.get("schema_hash") != schema_hash():
            raise ValueError("regression checkpoint representation schema is incompatible")
        if hyperparameters.get("architecture_hash") != architecture_hash(config):
            raise ValueError("regression checkpoint architecture is incompatible")
        if "target_mean" not in hyperparameters:
            raise ValueError(
                f"{resolved} is not a finetuned regression checkpoint "
                "(missing target_mean); pass --checkpoint pointing at a "
                "finetune experiment's best-*.ckpt"
            )
        module = RegressionModule(
            config,
            target_mean=float(hyperparameters["target_mean"]),
            target_std=float(hyperparameters["target_std"]),
        )
        state_dict = payload["state_dict"]
        ema_state_dict = payload.get("ema_state_dict")
        if isinstance(ema_state_dict, dict) and ema_state_dict:
            state_dict = {**state_dict, **ema_state_dict}
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        missing_set, unexpected_set = set(missing), set(unexpected)
        allowed_missing = _missing_gaussian_parameters(missing_set)
        if missing_set - allowed_missing:
            raise ValueError(f"regression checkpoint is missing parameters: {sorted(missing_set)}")
        if unexpected_set:
            raise ValueError(
                f"regression checkpoint has unexpected parameters: {sorted(unexpected_set)}"
            )
    finally:
        del payload
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device)
    module.eval()
    return module


def _missing_gaussian_parameters(missing: set[str]) -> set[str]:
    return {name for name in missing if name.startswith("model.uncertainty_head.")}
