from __future__ import annotations

import math

import torch

_SQRT_2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def gaussian_point_nll(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL for precise targets, all tensors in the same label space."""
    variances = sigmas.clamp_min(1e-6).square()
    per_item = 0.5 * ((targets - means).square() / variances + torch.log(variances)) + _LOG_SQRT_2PI
    if weights is None:
        weights = torch.ones_like(per_item)
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-8)


def gaussian_interval_nll(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log probability of a Gaussian falling inside an interval.

    Infinite bounds are fine (CDF endpoints clamp to 0/1); ``lower == upper``
    falls back to the point likelihood.
    """
    safe_sigmas = sigmas.clamp_min(1e-6)
    point = lower_bounds == upper_bounds
    finite_lower = torch.isfinite(lower_bounds)
    finite_upper = torch.isfinite(upper_bounds)
    safe_lower = torch.where(finite_lower, lower_bounds, torch.zeros_like(lower_bounds))
    safe_upper = torch.where(finite_upper, upper_bounds, torch.zeros_like(upper_bounds))
    lower_z = (safe_lower - means) / (safe_sigmas * _SQRT_2)
    upper_z = (safe_upper - means) / (safe_sigmas * _SQRT_2)
    lower_cdf = torch.where(
        finite_lower,
        0.5 * (1.0 + torch.erf(lower_z)),
        torch.zeros_like(lower_z),
    )
    upper_cdf = torch.where(
        finite_upper,
        0.5 * (1.0 + torch.erf(upper_z)),
        torch.ones_like(upper_z),
    )
    probability = (upper_cdf - lower_cdf).clamp_min(1e-12)
    interval_nll = -torch.log(probability)
    z = (safe_lower - means) / safe_sigmas
    point_nll = 0.5 * z.square() + torch.log(safe_sigmas) + _LOG_SQRT_2PI
    per_item = torch.where(point, point_nll, interval_nll)
    if weights is None:
        weights = torch.ones_like(per_item)
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-8)
