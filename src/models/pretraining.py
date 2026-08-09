from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.encoder import ChartEncoder
from preprocessing.representation import CATEGORICAL_FIELDS, CONTINUOUS_FIELDS

_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("note_type",),
    "geometry": (
        "key_id",
        "key_group",
        "slide_pattern",
        "slide_end",
    ),
    "timing": CONTINUOUS_FIELDS,
    "modifier": (
        "is_break",
        "is_fireworks",
        "is_ex",
    ),
}
# Flags are model inputs but omitted from reconstruction (near-constant, MLM loss would be noise).
_EXCLUDED_OBJECTIVE_FIELDS: frozenset[str] = frozenset()


class MaskedEventModel(nn.Module):
    def __init__(
        self,
        encoder: ChartEncoder,
        *,
        hidden_size: int,
        vocab_sizes: tuple[int, ...],
        continuous_fields: int,
        objective_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if len(vocab_sizes) != len(CATEGORICAL_FIELDS):
            raise ValueError("vocab_sizes must have one entry per categorical field")
        if continuous_fields != len(CONTINUOUS_FIELDS):
            raise ValueError("continuous_fields must match the representation schema")
        weights = {
            "core": 1.0,
            "geometry": 2.0,
            "timing": 1.0,
            "modifier": 0.25,
            **(objective_weights or {}),
        }
        if set(weights) != set(_FIELD_GROUPS) or any(weight <= 0 for weight in weights.values()):
            raise ValueError(
                "objective_weights must contain positive weights for every field group"
            )
        self.encoder = encoder
        self.objective_weights = weights
        self.categorical_heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in vocab_sizes)
        self.continuous_head = nn.Linear(hidden_size, continuous_fields)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden, _ = self.encoder(
            batch["categorical"],
            batch["continuous"],
            batch["continuous_presence"],
            batch["attention_mask"],
        )
        result = {
            field_name: head(hidden)
            for field_name, head in zip(CATEGORICAL_FIELDS, self.categorical_heads, strict=True)
        }
        result["continuous"] = self.continuous_head(hidden)
        return result

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Balanced reconstruction losses.

        ``token_loss`` is the plain all-target average, kept for comparison
        with earlier runs. ``objective_loss`` is the training objective:
        fields average inside semantic groups and groups get explicit
        weights, so plentiful/easy flags cannot drown out geometry or timing.
        """
        losses: dict[str, torch.Tensor] = {}
        zero = predictions["continuous"].sum() * 0.0
        token_loss_sum = zero
        total_count = torch.zeros((), dtype=torch.long, device=zero.device)
        field_means: dict[str, torch.Tensor] = {}
        field_counts: dict[str, torch.Tensor] = {}

        for field, field_name in enumerate(CATEGORICAL_FIELDS):
            name = field_name
            logits = predictions[name]
            labels = batch["categorical_labels"][:, :, field]
            valid = labels != -100
            count = valid.sum()
            field_sum = (
                F.cross_entropy(logits[valid], labels[valid], reduction="sum")
                if bool(count)
                else logits.sum() * 0.0
            )
            mean = field_sum / count.clamp_min(1).to(field_sum.dtype)
            losses[name] = mean
            losses[f"{name}_count"] = count
            field_means[field_name] = mean
            field_counts[field_name] = count
            token_loss_sum = token_loss_sum + field_sum
            total_count = total_count + count

        continuous_mask = batch["continuous_label_mask"]
        continuous_sum = zero
        continuous_count = torch.zeros((), dtype=torch.long, device=zero.device)
        for field, field_name in enumerate(CONTINUOUS_FIELDS):
            valid = continuous_mask[:, :, field]
            count = valid.sum()
            prediction = predictions["continuous"][:, :, field]
            label = batch["continuous_labels"][:, :, field]
            field_sum = (
                F.smooth_l1_loss(prediction[valid], label[valid], reduction="sum")
                if bool(count)
                else prediction.sum() * 0.0
            )
            mean = field_sum / count.clamp_min(1).to(field_sum.dtype)
            losses[field_name] = mean
            losses[f"{field_name}_count"] = count
            field_means[field_name] = mean
            field_counts[field_name] = count
            continuous_sum = continuous_sum + field_sum
            continuous_count = continuous_count + count

        losses["continuous"] = continuous_sum / continuous_count.clamp_min(1).to(zero.dtype)
        losses["continuous_count"] = continuous_count
        token_loss_sum = token_loss_sum + continuous_sum
        total_count = total_count + continuous_count

        objective_sum = zero
        active_weight = 0.0
        for group, fields in _FIELD_GROUPS.items():
            active_fields = [field for field in fields if bool(field_counts[field])]
            group_loss = (
                torch.stack([field_means[field] for field in active_fields]).mean()
                if active_fields
                else zero
            )
            group_count = sum((field_counts[field] for field in fields), start=total_count * 0)
            losses[f"{group}_loss"] = group_loss
            losses[f"{group}_count"] = group_count
            if active_fields:
                weight = self.objective_weights[group]
                objective_sum = objective_sum + weight * group_loss
                active_weight += weight

        # Keep empty heads in the autograd graph with zero gradient (DDP compatibility).
        objective_sum = objective_sum + torch.stack(tuple(field_means.values())).sum() * 0.0
        losses["target_count"] = total_count
        losses["token_loss"] = token_loss_sum / total_count.clamp_min(1).to(zero.dtype)
        losses["objective_loss"] = objective_sum / max(active_weight, 1.0)
        # The Lightning training step consumes this canonical key.
        losses["loss"] = losses["objective_loss"]
        return losses
