from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _is_hidden_block(module_name: str) -> bool:
    return ".transformer." in f".{module_name}" or ".aggregator.layers." in f".{module_name}"


def _muon_parameter_ids(module: nn.Module) -> set[int]:
    """Hidden Linear/MHA matrices eligible for Muon.

    Name-suffix checks miss ``MultiheadAttention.in_proj_weight`` and can
    catch LayerNorm scales; restricting by owning module type keeps
    embeddings, norms, biases, input projections and heads on AdamW.
    """
    result: set[int] = set()
    for name, child in module.named_modules():
        if not _is_hidden_block(name):
            continue
        if isinstance(child, nn.Linear):
            result.add(id(child.weight))
        elif isinstance(child, nn.MultiheadAttention) and child.in_proj_weight is not None:
            result.add(id(child.in_proj_weight))
    return result


def _partition_parameters(
    module: nn.Module,
    *,
    head_parameter_ids: set[int] | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    muon: list[nn.Parameter] = []
    encoder_adamw: list[nn.Parameter] = []
    head_adamw: list[nn.Parameter] = []
    head_parameter_ids = head_parameter_ids or set()
    muon_parameter_ids = _muon_parameter_ids(module)
    seen: set[int] = set()
    for _, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        if identifier in muon_parameter_ids:
            muon.append(parameter)
        elif identifier in head_parameter_ids:
            head_adamw.append(parameter)
        else:
            encoder_adamw.append(parameter)
    if not muon:
        raise ValueError("Muon hybrid optimizer found no eligible hidden matrix parameters")
    if not encoder_adamw and not head_adamw:
        raise ValueError("Muon hybrid optimizer found no AdamW-compatible parameters")
    return muon, encoder_adamw, head_adamw


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Single-optimizer facade over ``torch.optim.Muon`` plus auxiliary AdamW.

    Lightning only supports one optimizer under automatic optimization, so
    this class presents all parameter groups as a single optimizer while
    delegating the actual updates to the built-in ``torch.optim.Muon``
    (hidden matrices) and ``torch.optim.AdamW`` (embeddings, norms, biases,
    heads). LR scheduling, gradient clipping, checkpointing, and DDP all
    work exactly as with a plain optimizer; learning rates are propagated
    from the outer param groups to the inner optimizers before each step.

    ``state_dict``/``load_state_dict`` delegate to the inner optimizers, so
    momentum buffers and Adam moments survive checkpoints.
    """

    def __init__(
        self,
        param_groups: list[dict[str, object]],
        *,
        muon_lr: float,
        muon_momentum: float,
        muon_adjust_lr: str,
        weight_decay: float,
        beta1: float,
        beta2: float,
        adam_epsilon: float,
    ) -> None:
        super().__init__(param_groups, {})
        muon_parameters: list[nn.Parameter] = []
        adamw_groups: list[dict[str, object]] = []
        for group in self.param_groups:
            if group["use_muon"]:
                muon_parameters.extend(group["params"])
            else:
                adamw_groups.append(group)
        if not muon_parameters:
            raise ValueError("MuonWithAuxAdam requires at least one Muon parameter group")
        for parameter in muon_parameters:
            if parameter.ndim != 2:
                raise ValueError(
                    "torch.optim.Muon only supports 2D parameters, got "
                    f"{parameter.ndim}D with shape {tuple(parameter.shape)}"
                )
        self._muon = torch.optim.Muon(
            muon_parameters,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=weight_decay,
            nesterov=True,
            adjust_lr_fn=muon_adjust_lr,
        )
        self._adamw = torch.optim.AdamW(
            adamw_groups,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=adam_epsilon,
        )
        self._sync_learning_rates()

    @property
    def muon(self) -> torch.optim.Muon:
        """The inner ``torch.optim.Muon`` (for diagnostics/tests)."""
        return self._muon

    def _sync_learning_rates(self) -> None:
        """Propagate scheduler-driven LR changes to the inner optimizers."""
        self._muon.param_groups[0]["lr"] = self.param_groups[0]["lr"]
        adamw_groups = [group for group in self.param_groups if not group["use_muon"]]
        for outer, inner in zip(adamw_groups, self._adamw.param_groups, strict=True):
            inner["lr"] = outer["lr"]

    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sync_learning_rates()
        if any(p.grad is not None for group in self._muon.param_groups for p in group["params"]):
            self._muon.step()  # type: ignore[no-untyped-call]
        if any(p.grad is not None for group in self._adamw.param_groups for p in group["params"]):
            self._adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "muon": self._muon.state_dict(),
            "adamw": self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self._sync_learning_rates()


def muon_parameter_names(module: nn.Module) -> set[str]:
    """Expose the partition for tests and diagnostic output."""
    parameter_ids = _muon_parameter_ids(module)
    return {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and id(parameter) in parameter_ids
    }
