from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def transfer_encoder(
    encoder: nn.Module,
    checkpoint: str | Path,
    *,
    expected_schema_hash: str,
    expected_architecture_hash: str,
    strict_architecture: bool = True,
) -> None:
    payload: dict[str, Any] = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    hyperparameters = payload.get("hyper_parameters", {})
    actual_schema = hyperparameters.get("schema_hash")
    actual_architecture = hyperparameters.get("architecture_hash")
    if actual_schema != expected_schema_hash:
        raise ValueError(
            f"representation schema mismatch: checkpoint={actual_schema}, "
            f"expected={expected_schema_hash}"
        )
    if strict_architecture and actual_architecture != expected_architecture_hash:
        raise ValueError(
            f"encoder architecture mismatch: checkpoint={actual_architecture}, "
            f"expected={expected_architecture_hash}"
        )

    state_dict = payload.get("state_dict", payload)
    prefixes = ("model.encoder.", "encoder.")
    encoder_state: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                encoder_state[name.removeprefix(prefix)] = value
                break
    if not encoder_state:
        raise ValueError("checkpoint contains no encoder parameters")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
    if missing or unexpected:
        raise ValueError(f"encoder transfer mismatch: missing={missing}, unexpected={unexpected}")
