"""Depth-weighted masked MSE and the GREP auxiliary mix (PRD_AI_ML §5.4).

    L(Y) = sum(w_d * M * (Yhat - Y)^2) / sum(w_d * M)

Cells with M = False are ignored, and their target values are never read, so NaN in the
target does not reach the loss. The total is

    L = (1 - a) * L(Y_glorys) + a * L(Y_grep)

where a = aux_weight. The GREP term uses only the samples in the batch where GREP exists. If no
sample in the batch has GREP, the total is L(Y_glorys).
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


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    depth_weights: torch.Tensor,
) -> torch.Tensor:
    """Depth-weighted MSE over valid cells. pred, target: [B, D, H, W]; mask: bool same shape."""
    weights = depth_weights.view(1, -1, 1, 1).to(pred.dtype) * mask.to(pred.dtype)
    # Masked targets are replaced first, so NaN there cannot leak into the value or the gradient.
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    squared = torch.where(mask, (pred - safe_target) ** 2, torch.zeros_like(pred))
    denominator = weights.sum()
    if float(denominator) == 0.0:
        return pred.sum() * 0.0
    return (weights * squared).sum() / denominator


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    grep_target: torch.Tensor,
    grep_mask: torch.Tensor,
    has_grep: torch.Tensor,
    depth_weights: torch.Tensor,
    aux_weight: float,
) -> torch.Tensor:
    """GLORYS loss, mixed with the GREP loss on samples where GREP exists (§5.4)."""
    glorys = masked_weighted_mse(pred, target, mask, depth_weights)
    if not bool(has_grep.any()):
        return glorys
    grep_cells = grep_mask & has_grep.view(-1, 1, 1, 1)
    grep = masked_weighted_mse(pred, grep_target, grep_cells, depth_weights)
    return (1.0 - aux_weight) * glorys + aux_weight * grep
