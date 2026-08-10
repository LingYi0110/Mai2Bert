from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import nn

from lib.config.schema import AppConfig


class EmaTracker:
    def __init__(self, parameters: dict[str, nn.Parameter], decay: float) -> None:
        self.decay = decay
        self._parameters = dict(parameters)
        self._shadow = {
            name: parameter.detach().clone() for name, parameter in self._parameters.items()
        }
        self._backup: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._shadow)

    def step(self) -> None:
        for name, parameter in self._parameters.items():
            self._shadow[name] = (
                self.decay * self._shadow[name] + (1.0 - self.decay) * parameter.data
            ).clone()

    def apply(self) -> None:
        for name, parameter in self._parameters.items():
            self._backup[name] = parameter.data
            parameter.data = self._shadow[name]

    def restore(self) -> None:
        for name, tensor in self._backup.items():
            self._parameters[name].data = tensor
        self._backup.clear()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self._shadow.items()}

    def to_(self, device: torch.device) -> None:
        """Move the shadow copy onto ``device``.

        Checkpoint serialization drops tensor device information (pickle
        restores CPU tensors), so after ``load_state_dict`` the shadow may
        live on a different device than the live model parameters.
        """
        if any(tensor.device != device for tensor in self._shadow.values()):
            self._shadow = {
                name: tensor.to(device) for name, tensor in self._shadow.items()
            }

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        for name, tensor in self._shadow.items():
            if tensor.shape != state_dict[name].shape:
                raise RuntimeError(
                    f"shape mismatch for '{name}': expected {tensor.shape}, "
                    f"got {state_dict[name].shape}"
                )
            self._shadow[name] = state_dict[name].clone()


class EmaMixin(L.LightningModule):
    config: AppConfig

    def _ensure_ema(self) -> None:
        decay = self.config.optimizer.ema_decay
        if decay is None or hasattr(self, "_ema"):
            return
        self._ema = EmaTracker(dict(self.named_parameters()), decay=decay)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self._ensure_ema()
        if hasattr(self, "_ema"):
            # Pickle drops tensor devices; realign the shadow after Lightning
            # moved the model onto its final device (before any val/step hook
            # touches the EMA copy).
            self._ema.to_(next(self.parameters()).device)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        self._ensure_ema()
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is not None:
            self._ema.load_state_dict(ema_state)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if hasattr(self, "_ema"):
            checkpoint["ema_state_dict"] = self._ema.state_dict()

    def optimizer_step(self, *args: Any, **kwargs: Any) -> Any:
        result = super().optimizer_step(*args, **kwargs)
        if hasattr(self, "_ema"):
            self._ema.step()
        return result

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if hasattr(self, "_ema"):
            self._ema.apply()

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "_ema"):
            self._ema.restore()
        super().on_validation_epoch_end()
