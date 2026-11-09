"""Huber and vertical-gradient loss options (models.loss); defaults leave the loss unchanged."""

from __future__ import annotations

import pytest
import torch

from oceanscope_ml.models.loss import masked_weighted_mse, total_loss, vertical_gradient_mse


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    pred = torch.randn(2, 4, 3, 5, generator=gen)
    target = torch.randn(2, 4, 3, 5, generator=gen)
    mask = torch.rand(2, 4, 3, 5, generator=gen) > 0.3
    weights = torch.tensor([1.0, 2.0, 2.0, 1.0])
    return pred, target, mask, weights


def test_huber_with_a_large_delta_equals_mse() -> None:
    pred, target, mask, w = _batch()
    assert torch.allclose(
        masked_weighted_mse(pred, target, mask, w, huber_delta=1e6),
        masked_weighted_mse(pred, target, mask, w),
    )


def test_huber_pulls_less_on_large_errors() -> None:
    pred, target, mask, w = _batch()
    target = target.clone()
    target[0, 0, 0, 0] = 50.0
    mask[0, 0, 0, 0] = True
    assert masked_weighted_mse(pred, target, mask, w, huber_delta=1.0) < masked_weighted_mse(
        pred, target, mask, w
    )


def test_huber_ignores_masked_nan() -> None:
    pred, target, mask, w = _batch()
    target[~mask] = float("nan")
    pred = pred.clone().requires_grad_(True)
    loss = masked_weighted_mse(pred, target, mask, w, huber_delta=0.5)
    loss.backward()
    assert torch.isfinite(loss) and pred.grad is not None and torch.isfinite(pred.grad).all()


def test_vertical_gradient_is_zero_for_a_uniform_offset_in_degrees() -> None:
    _, target, mask, _ = _batch()
    sigma = torch.tensor([0.5, 1.0, 2.0, 0.8])
    pred = target + 0.7 / sigma.view(1, -1, 1, 1)  # +0.7 C at every level
    assert float(vertical_gradient_mse(pred, target, mask, sigma)) == pytest.approx(0.0, abs=1e-9)
    assert float(vertical_gradient_mse(target * 2, target, mask, sigma)) > 0


def test_vertical_gradient_uses_only_valid_pairs_and_ignores_nan() -> None:
    pred, target, mask, _ = _batch()
    sigma = torch.ones(4)
    mask = torch.zeros_like(mask)
    mask[:, 0] = True  # no neighbouring pair is valid
    target[~mask] = float("nan")
    assert float(vertical_gradient_mse(pred, target, mask, sigma)) == 0.0


def test_total_loss_defaults_unchanged_and_vgrad_needs_sigma() -> None:
    pred, target, mask, w = _batch()
    has = torch.tensor([True, False])
    base = total_loss(pred, target, mask, target, mask, has, w, 0.5)
    same = total_loss(pred, target, mask, target, mask, has, w, 0.5, None, 0.0, None)
    assert torch.equal(base, same)
    with pytest.raises(ValueError):
        total_loss(pred, target, mask, target, mask, has, w, 0.5, vgrad_weight=0.1)
    more = total_loss(
        pred, target, mask, target, mask, has, w, 0.5, vgrad_weight=0.1, sigma_d=torch.ones(4)
    )
    assert more > base
