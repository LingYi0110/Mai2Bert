from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from models.embedding import StructuredEventEmbedding


class StochasticDepth(nn.Module):
    """Drop a sub-layer's output with probability p during training.

    Drop probability ramps linearly with depth; at eval time the layer
    always runs.
    """

    def __init__(self, module: nn.Module, drop_prob: float) -> None:
        super().__init__()
        self.module = module
        self.drop_prob = float(drop_prob)

    def forward(self, src: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0:
            return cast(torch.Tensor, self.module(src, *args, **kwargs))
        mask = torch.rand((src.shape[0], 1, 1), device=src.device) >= self.drop_prob
        out = cast(torch.Tensor, self.module(src, *args, **kwargs))
        return torch.where(mask, out, src)


class ChartEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_sizes: tuple[int, ...],
        continuous_fields: int,
        hidden_size: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        feedforward_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_events: int = 512,
        stochastic_depth: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_events = max_events
        self.embedding = StructuredEventEmbedding(
            vocab_sizes,
            continuous_fields,
            hidden_size,
            max_events,
            dropout,
        )
        layers: list[nn.Module] = []
        for index in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            if stochastic_depth > 0 and num_layers > 1:
                drop_prob = stochastic_depth * index / (num_layers - 1)
                layers.append(StochasticDepth(layer, drop_prob))
            else:
                layers.append(layer)
        self.transformer = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.pool_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        continuous_presence: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(categorical, continuous, continuous_presence)
        cls_mask = torch.ones(
            (attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device
        )
        full_valid_mask = torch.cat((cls_mask, attention_mask), dim=1)
        hidden = self._run_layers(embedded, ~full_valid_mask)
        event_hidden = hidden[:, 1:]
        denominator = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        event_mean = (event_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / denominator
        pooled = self.pool_norm(hidden[:, 0] + event_mean)
        return event_hidden, pooled

    def _run_layers(self, src: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Run wrapped layers in sequence, propagating the padding mask.

        ``nn.TransformerEncoder`` passes ``src_key_padding_mask`` to any real
        ``TransformerEncoderLayer``; the wrapper forwards it too, and skipped
        layers consume it identically.
        """
        hidden = src
        for layer in self.transformer:
            hidden = cast(
                torch.Tensor,
                layer(hidden, src_key_padding_mask=key_padding_mask),
            )
        return cast(torch.Tensor, self.final_norm(hidden))
