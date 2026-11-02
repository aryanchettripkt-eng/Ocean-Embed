"""Masked, depth-weighted MSE and the GREP mix (PRD_AI_ML §5.4)."""

from __future__ import annotations

import torch

from oceanscope_ml.models.loss import masked_weighted_mse, total_loss


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    pred = torch.randn(2, 3, 4, 5, generator=gen)
    target = torch.randn(2, 3, 4, 5, generator=gen)
    mask = torch.rand(2, 3, 4, 5, generator=gen) > 0.3
    weights = torch.tensor([1.0, 2.0, 3.0])
    return pred, target, mask, weights


def test_masked_nan_does_not_reach_value_or_gradient() -> None:
    pred, target, mask, weights = _batch()
    target[~mask] = float("nan")
    pred = pred.clone().requires_grad_(True)

    loss = masked_weighted_mse(pred, target, mask, weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_matches_weighted_mean_over_valid_cells() -> None:
    pred, target, mask, weights = _batch()
    target[~mask] = float("nan")

    w_full = weights.view(1, -1, 1, 1).expand_as(pred)[mask]
    sq_err = (pred[mask] - target[mask]) ** 2
    expected = (w_full * sq_err).sum() / w_full.sum()

    loss = masked_weighted_mse(pred, target, mask, weights)
    assert torch.allclose(loss, expected)


def test_no_valid_cells_gives_zero_loss_and_zero_gradient() -> None:
    pred = torch.randn(1, 2, 3, 3, requires_grad=True)
    target = torch.full_like(pred, float("nan"))
    mask = torch.zeros_like(pred, dtype=torch.bool)

    loss = masked_weighted_mse(pred, target, mask, torch.ones(2))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_grep_term_uses_only_samples_that_have_grep() -> None:
    pred, target, mask, weights = _batch(seed=1)
    grep_target = target + 1.0
    grep_mask = mask.clone()
    has_grep = torch.tensor([True, False])

    glorys = masked_weighted_mse(pred, target, mask, weights)
    grep_cells = grep_mask & has_grep.view(-1, 1, 1, 1)
    grep = masked_weighted_mse(pred, grep_target, grep_cells, weights)

    mixed = total_loss(pred, target, mask, grep_target, grep_mask, has_grep, weights, 0.5)
    assert torch.allclose(mixed, 0.5 * glorys + 0.5 * grep)


def test_total_loss_is_glorys_only_when_no_sample_has_grep() -> None:
    pred, target, mask, weights = _batch(seed=2)
    no_grep = torch.zeros(2, dtype=torch.bool)
    glorys = masked_weighted_mse(pred, target, mask, weights)

    loss = total_loss(pred, target, mask, target, mask, no_grep, weights, 0.5)
    assert torch.allclose(loss, glorys)
