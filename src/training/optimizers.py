from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from lib.config.schema import OptimizerConfig
from lib.optimizer.hybrid_muon import MuonWithAuxAdam, _partition_parameters


def build_optimizer(
    module: nn.Module,
    config: OptimizerConfig,
    *,
    head_parameters: Iterable[nn.Parameter] = (),
) -> Optimizer:
    """Build AdamW, or one hybrid Muon/AdamW optimizer.

    A single optimizer keeps Lightning scheduling/checkpointing simple. Muon
    touches only hidden Transformer/attention matrices; embeddings, norms,
    biases and heads stay on AdamW (Keller Jordan hybrid design).
    """
    head_parameter_ids = {id(parameter) for parameter in head_parameters}

    if config.name == "adamw":
        encoder_parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in head_parameter_ids
        ]
        head_parameters_list = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) in head_parameter_ids
        ]
        groups: list[dict[str, object]] = [
            {"params": encoder_parameters, "lr": config.encoder_lr},
        ]
        if head_parameters_list:
            groups.append({"params": head_parameters_list, "lr": config.head_lr})
        return torch.optim.AdamW(
            groups,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon,
        )

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "muon_hybrid requires torch>=2.9 with torch.optim.Muon "
            f"(installed: torch {torch.__version__})"
        )

    muon_parameters, encoder_adamw_parameters, head_adamw_parameters = _partition_parameters(
        module,
        head_parameter_ids=head_parameter_ids,
    )
    param_groups: list[dict[str, object]] = [
        {
            "params": muon_parameters,
            "lr": config.muon_lr,
            "use_muon": True,
        },
        {
            "params": encoder_adamw_parameters,
            "lr": config.encoder_lr,
            "use_muon": False,
        },
    ]
    if head_adamw_parameters:
        param_groups.append(
            {
                "params": head_adamw_parameters,
                "lr": config.head_lr,
                "use_muon": False,
            }
        )
    return MuonWithAuxAdam(
        param_groups,
        muon_lr=config.muon_lr,
        muon_momentum=config.muon_momentum,
        muon_adjust_lr=config.muon_adjust_lr,
        weight_decay=config.weight_decay,
        beta1=config.beta1,
        beta2=config.beta2,
        adam_epsilon=config.adam_epsilon,
    )
