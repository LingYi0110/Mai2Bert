from __future__ import annotations

import torch
from torch import nn


class StructuredEventEmbedding(nn.Module):
    def __init__(
        self,
        vocab_sizes: tuple[int, ...],
        continuous_fields: int,
        hidden_size: int,
        max_events: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_size, padding_idx=0) for size in vocab_sizes
        )
        self.continuous_projection = nn.Sequential(
            nn.Linear(continuous_fields * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position_embedding = nn.Embedding(max_events + 1, hidden_size)
        self.cls = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        continuous_presence: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, fields = categorical.shape
        if fields != len(self.categorical_embeddings):
            raise ValueError(
                f"expected {len(self.categorical_embeddings)} categorical fields, got {fields}"
            )
        if length + 1 > self.position_embedding.num_embeddings:
            raise ValueError(
                f"sequence length {length} exceeds encoder maximum "
                f"{self.position_embedding.num_embeddings - 1}"
            )
        embedded = torch.zeros(
            (batch_size, length, self.cls.shape[-1]),
            device=categorical.device,
            dtype=self.cls.dtype,
        )
        for field, embedding in enumerate(self.categorical_embeddings):
            embedded = embedded + embedding(categorical[:, :, field])
        numeric = torch.cat((continuous, continuous_presence.to(continuous.dtype)), dim=-1)
        embedded = embedded + self.continuous_projection(numeric)
        embedded = torch.cat((self.cls.expand(batch_size, -1, -1), embedded), dim=1)
        positions = torch.arange(length + 1, device=categorical.device)
        embedded = embedded + self.position_embedding(positions).unsqueeze(0)
        result: torch.Tensor = self.dropout(self.layer_norm(embedded))
        return result
