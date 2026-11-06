"""Depth-weighted masked MSE and the GREP auxiliary mix (PRD_AI_ML §5.4).

    L(Y) = sum(w_d * M * (Yhat - Y)^2) / sum(w_d * M)

Cells with M = False are ignored, and their target values are never read, so NaN in the
target does not reach the loss. The total is

    L = (1 - a) * L(Y_glorys) + a * L(Y_grep)

where a = aux_weight. The GREP term uses only the samples in the batch where GREP exists. If no
sample in the batch has GREP, the total is L(Y_glorys).

Two optional terms for experiments (both off by default, so existing runs are unchanged):

- ``huber_delta``: the squared error is replaced by the Huber loss with that threshold (in
  normalized units), so a few large errors (fronts, bad profiles) pull less on the fit.
- ``vgrad_weight``: adds ``vgrad_weight * L_vgrad``, the masked MSE of the change between
  neighbouring depth levels, in degrees C scaled by the mean sigma of the two levels. It
  penalises a wrong vertical temperature gradient (the thermocline shape) even where the
  level values are close.
"""

from __future__ import annotations

import torch

from oceanscope_ml.config import ExperimentConfig


def depth_weight_vector(config: ExperimentConfig) -> torch.Tensor:
    """Per-depth weights w_d from target.depth_weights (in_band inside band_m, else out_band)."""
    weights_cfg = config.target.get("depth_weights", {})
    lo, hi = (float(v) for v in weights_cfg.get("band_m", [50.0, 300.0]))
    in_band = float(weights_cfg.get("in_band", 1.0))
    out_band = float(weights_cfg.get("out_band", 1.0))
    values = [in_band if lo <= d <= hi else out_band for d in config.depths_m]
    return torch.tensor(values, dtype=torch.float32)


def _pointwise(diff: torch.Tensor, huber_delta: float | None) -> torch.Tensor:
    """Squared error, or the Huber loss (0.5 r^2 inside delta, delta (|r| - delta/2) beyond),
    scaled by 2 so both agree for small errors."""
    if huber_delta is None:
        return diff**2
    a = diff.abs()
    delta = float(huber_delta)
    return 2.0 * torch.where(a <= delta, 0.5 * diff**2, delta * (a - 0.5 * delta))


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depth_weights: torch.Tensor,
    huber_delta: float | None = None,
) -> torch.Tensor:
    """Depth-weighted MSE (or Huber with ``huber_delta``) over valid cells.
    pred, target: [B, D, H, W]; mask: bool same shape."""
    weights = depth_weights.view(1, -1, 1, 1).to(pred.dtype) * mask.to(pred.dtype)
    # Masked targets are replaced first, so NaN there cannot leak into the value or the gradient.
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    squared = torch.where(mask, _pointwise(pred - safe_target, huber_delta), torch.zeros_like(pred))
    denominator = weights.sum()
    if float(denominator) == 0.0:
        return pred.sum() * 0.0
    return (weights * squared).sum() / denominator


def vertical_gradient_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sigma_d: torch.Tensor,
) -> torch.Tensor:
    """MSE of the level-to-level change, on pairs of neighbouring levels both valid.

    pred and target are normalized anomalies (value / sigma_d). The change is taken in degrees C
    (times sigma_d) and divided by the mean sigma of the two levels, so it is on the same scale
    as the main loss. Returns 0 when no pair is valid.
    """
    sigma = sigma_d.view(1, -1, 1, 1).to(pred.dtype)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    p, t = pred * sigma, safe_target * sigma
    pair = mask[:, 1:] & mask[:, :-1]
    scale = 0.5 * (sigma[:, 1:] + sigma[:, :-1])
    diff = ((p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])) / scale
    squared = torch.where(pair, diff**2, torch.zeros_like(diff))
    n = pair.sum()
    if int(n) == 0:
        return pred.sum() * 0.0
    return squared.sum() / n


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grep_target: torch.Tensor,
    grep_mask: torch.Tensor,
    has_grep: torch.Tensor,
    depth_weights: torch.Tensor,
    aux_weight: float,
    huber_delta: float | None = None,
    vgrad_weight: float = 0.0,
    sigma_d: torch.Tensor | None = None,
) -> torch.Tensor:
    """GLORYS loss, mixed with the GREP loss on samples where GREP exists (§5.4), plus the
    optional vertical-gradient term on GLORYS (needs ``sigma_d``)."""
    glorys = masked_weighted_mse(pred, target, mask, depth_weights, huber_delta)
    if vgrad_weight > 0:
        if sigma_d is None:
            raise ValueError("vgrad_weight > 0 needs sigma_d")
        glorys = glorys + vgrad_weight * vertical_gradient_mse(pred, target, mask, sigma_d)
    if not bool(has_grep.any()):
        return glorys
    grep_cells = grep_mask & has_grep.view(-1, 1, 1, 1)
    grep = masked_weighted_mse(pred, grep_target, grep_cells, depth_weights, huber_delta)
    return (1.0 - aux_weight) * glorys + aux_weight * grep
