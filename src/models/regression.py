from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from data.batching import compound_window_ranges
from models.encoder import ChartEncoder


def _inverse_softplus(value: float) -> float:
    return float(torch.log(torch.expm1(torch.tensor(float(value)))).item())


class WindowAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = inputs + attended
        result: torch.Tensor = hidden + self.ffn(self.ffn_norm(hidden))
        return result


class WindowAttentionAggregator(nn.Module):
    """Attention over per-window pools with a learned query.

    A learned query is prepended to the window vectors, a small stack of
    pre-norm Transformer blocks attends over them, and the final query
    position is read out as the chart embedding. Each window gets an explicit
    normalized ``(start, end)`` position encoding so coarse chart order and
    coverage survive the variable window count.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int,
        num_layers: int = 1,
        feedforward_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        if hidden_size % num_heads:
            raise ValueError("aggregator_heads must divide hidden_size")
        feedforward = feedforward_size if feedforward_size is not None else 2 * hidden_size
        self.query = nn.Parameter(torch.empty(1, 1, hidden_size))
        # Project normalized (start, end) into window space for position/coverage.
        self.position_projection = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList(
            WindowAttentionBlock(
                hidden_size,
                num_heads=num_heads,
                feedforward_size=feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        window_embeds: torch.Tensor,
        window_positions: torch.Tensor | None = None,
        window_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate one chart, or a padded batch (via ``window_padding_mask``)."""
        if window_embeds.ndim == 3:
            return self._forward_batch(window_embeds, window_positions, window_padding_mask)
        if window_embeds.ndim != 2:
            raise ValueError(
                "window_embeds must have shape (windows, hidden) or (batch, windows, hidden)"
            )
        if window_embeds.shape[0] == 0:
            raise ValueError("window_embeds cannot be empty")
        if window_positions is None:
            denominator = float(window_embeds.shape[0])
            starts = (
                torch.arange(
                    window_embeds.shape[0], device=window_embeds.device, dtype=window_embeds.dtype
                )
                / denominator
            )
            ends = (starts + 1.0 / denominator).clamp(max=1.0)
            window_positions = torch.stack((starts, ends), dim=-1)
        if window_positions.shape != (window_embeds.shape[0], 2):
            raise ValueError("window_positions must have shape (windows, 2)")
        position_features = self.position_projection(window_positions.to(window_embeds.dtype))
        positioned = window_embeds + position_features
        # x: (1, 1 + n_windows, hidden) with query at position 0
        x = torch.cat(
            (
                self.query.to(window_embeds.dtype),
                positioned.unsqueeze(0),
            ),
            dim=1,
        )
        for layer in self.layers:
            x = layer(x)
        pooled = cast(torch.Tensor, self.final_norm(x[:, 0].squeeze(0)))
        return pooled

    def _forward_batch(
        self,
        window_embeds: torch.Tensor,
        window_positions: torch.Tensor | None,
        window_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, window_count, _ = window_embeds.shape
        if window_count == 0:
            raise ValueError("window_embeds cannot have zero windows")
        if window_positions is None:
            positions = torch.arange(
                window_count,
                device=window_embeds.device,
                dtype=window_embeds.dtype,
            )
            starts = positions / float(window_count)
            window_positions = torch.stack(
                (starts, (starts + 1.0 / window_count).clamp(max=1.0)), dim=-1
            ).expand(batch_size, -1, -1)
        if window_positions.shape != (batch_size, window_count, 2):
            raise ValueError("window_positions must have shape (batch, windows, 2)")
        if window_padding_mask is None:
            window_padding_mask = torch.zeros(
                (batch_size, window_count), dtype=torch.bool, device=window_embeds.device
            )
        if window_padding_mask.shape != (batch_size, window_count):
            raise ValueError("window_padding_mask must have shape (batch, windows)")
        position_features = self.position_projection(window_positions.to(window_embeds.dtype))
        positioned = window_embeds + position_features
        x = torch.cat(
            (
                self.query.to(window_embeds.dtype).expand(batch_size, -1, -1),
                positioned,
            ),
            dim=1,
        )
        padding = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=window_embeds.device),
                window_padding_mask,
            ),
            dim=1,
        )
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding)
        return cast(torch.Tensor, self.final_norm(x[:, 0]))


def chart_embeddings(
    encoder: ChartEncoder,
    aggregator: WindowAttentionAggregator,
    batch: dict[str, torch.Tensor],
    *,
    max_events: int,
    stride: int,
) -> torch.Tensor:
    """Encode one chart per batch row into a fixed-size embedding.

    Shared by every supervised task head. Charts longer than ``max_events``
    are split into compound-aligned windows inside the encoder; the
    aggregator pools them.
    """
    raw_window_ranges = batch.get("window_ranges")
    window_specs: list[tuple[int, int, int]] = []
    windows_per_chart: list[int] = []
    if raw_window_ranges is not None:
        for row, ranges in enumerate(raw_window_ranges):
            if not ranges:
                raise ValueError("each regression chart must contain at least one window")
            windows_per_chart.append(len(ranges))
            window_specs.extend((row, int(start), int(end)) for start, end in ranges)
    else:
        raw_lengths = batch.get("lengths")
        if raw_lengths is None:
            # Fallback for batches built without collate_events.
            lengths: list[int] = [
                int(value) for value in batch["attention_mask"].sum(dim=1).cpu().tolist()
            ]
        else:
            lengths = [int(value) for value in raw_lengths]
        for row, length in enumerate(lengths):
            if "note_start" not in batch:
                raise KeyError("regression batches must contain note_start")
            ranges = compound_window_ranges(
                batch["note_start"][row],
                length,
                max_events,
                stride,
            )
            windows_per_chart.append(len(ranges))
            for start, end in ranges:
                window_specs.append((row, start, end))

    max_window_length = max(end - start for _, start, end in window_specs)
    categorical = batch["categorical"].new_zeros(
        (len(window_specs), max_window_length, batch["categorical"].shape[-1])
    )
    continuous = batch["continuous"].new_zeros(
        (len(window_specs), max_window_length, batch["continuous"].shape[-1])
    )
    continuous_presence = batch["continuous_presence"].new_zeros(
        (len(window_specs), max_window_length, batch["continuous_presence"].shape[-1])
    )
    attention_mask = batch["attention_mask"].new_zeros((len(window_specs), max_window_length))
    for window, (row, start, end) in enumerate(window_specs):
        length = end - start
        categorical[window, :length] = batch["categorical"][row, start:end]
        continuous[window, :length] = batch["continuous"][row, start:end]
        continuous_presence[window, :length] = batch["continuous_presence"][row, start:end]
        attention_mask[window, :length] = batch["attention_mask"][row, start:end]

    if any(parameter.requires_grad for parameter in encoder.parameters()):
        _, pooled_windows = encoder(
            categorical,
            continuous,
            continuous_presence,
            attention_mask,
        )
    else:
        with torch.no_grad():
            _, pooled_windows = encoder(
                categorical,
                continuous,
                continuous_presence,
                attention_mask,
            )
    batch_size = len(windows_per_chart)
    max_windows = max(windows_per_chart)
    window_batch = pooled_windows.new_zeros((batch_size, max_windows, pooled_windows.shape[-1]))
    window_positions = pooled_windows.new_zeros((batch_size, max_windows, 2))
    window_padding_mask = torch.ones(
        (batch_size, max_windows), dtype=torch.bool, device=pooled_windows.device
    )
    offset = 0
    for row, count in enumerate(windows_per_chart):
        window_batch[row, :count] = pooled_windows[offset : offset + count]
        positions = torch.arange(
            count, device=pooled_windows.device, dtype=pooled_windows.dtype
        ) / float(count)
        window_positions[row, :count, 0] = positions
        window_positions[row, :count, 1] = (positions + 1.0 / count).clamp(max=1.0)
        window_padding_mask[row, :count] = False
        offset += count
    chart_embeddings = aggregator(
        window_batch,
        window_positions,
        window_padding_mask,
    )
    return cast(torch.Tensor, chart_embeddings)


class DifficultyRegressor(nn.Module):
    def __init__(
        self,
        encoder: ChartEncoder,
        *,
        hidden_size: int,
        regression_hidden_size: int,
        max_events: int = 512,
        stride: int = 384,
        dropout: float = 0.1,
        aggregator_heads: int | None = 1,
        aggregator_layers: int = 1,
        init_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.max_events = max_events
        self.stride = stride
        self.aggregator = WindowAttentionAggregator(
            hidden_size,
            num_heads=aggregator_heads or 1,
            num_layers=aggregator_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, regression_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_size, 1),
        )
        # Separate output so checkpoints gain it without disturbing the trained mean head.
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, regression_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_size, 1),
        )
        with torch.no_grad():
            final = cast(nn.Linear, self.uncertainty_head[-1])
            final.weight.zero_()
            final.bias.fill_(_inverse_softplus(init_sigma))

    def forward_gaussian(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized mean and positive sigma for Gaussian NLL."""
        embeddings = self._chart_embeddings(batch)
        mean: torch.Tensor = self.head(embeddings).squeeze(-1)
        log_sigma: torch.Tensor = self.uncertainty_head(embeddings).squeeze(-1)
        sigma = F.softplus(log_sigma).clamp_min(1e-4)
        return mean, sigma

    def _chart_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return chart_embeddings(
            self.encoder,
            self.aggregator,
            batch,
            max_events=self.max_events,
            stride=self.stride,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        chart_embeddings = self._chart_embeddings(batch)
        result: torch.Tensor = self.head(chart_embeddings).squeeze(-1)
        return result
