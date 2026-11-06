"""Correction guards, dev/final float halves and ensemble spread-skill, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oceanscope_ml.evaluation import spread_skill as ss
from oceanscope_ml.evaluation.argo_validation import holdout_halves
from oceanscope_ml.inference import argo_correct as ac


def _res(values: np.ndarray) -> ac.Residuals:
    n = values.shape[0]
    return ac.Residuals(
        np.array([f"f{k % 7}" for k in range(n)]),
        np.arange(n, dtype=np.int64),
        np.full(n, 10.0),
        np.full(n, 85.0),
        values,
    )


def test_drop_shallow_blanks_levels_above_the_limit() -> None:
    depths = np.array([0.5, 5.0, 50.0])
    res = _res(np.ones((4, 3)))
    out = ac.drop_shallow(res, depths, 1.0)
    assert np.isnan(out.values[:, 0]).all()
    assert np.isfinite(out.values[:, 1:]).all()
    only_top = _res(np.array([[1.0, np.nan, np.nan]]))
    assert len(ac.drop_shallow(only_top, depths, 1.0)) == 0


def test_drop_outliers_removes_only_far_values() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 0.2, size=(200, 2))
    values[5, 0] = 8.0
    values[9, 1] = -6.0
    out, dropped = ac.drop_outliers(_res(values), 4.0)
    assert dropped.tolist() == [1, 1]
    assert np.nanmax(np.abs(out.values)) < 2.0
    assert len(out) == 200  # each row still has its other level


def test_holdout_halves_are_disjoint_stratified_and_reproducible() -> None:
    rows = []
    for k in range(15):  # Bay of Bengal
        rows.append((f"b{k}", 12.0, 88.0))
    for k in range(10):  # Arabian Sea
        rows.append((f"a{k}", 15.0, 65.0))
    for k in range(5):  # equatorial
        rows.append((f"e{k}", 2.0, 80.0))
    profiles = pd.DataFrame(rows * 3, columns=["platform_number", "lat", "lon"])
    a, b = holdout_halves(profiles), holdout_halves(profiles)
    pd.testing.assert_frame_equal(a, b)
    counts = a.groupby(["region", "half"]).size().to_dict()
    assert counts[("bay_of_bengal", "dev")] == 8 and counts[("bay_of_bengal", "final")] == 7
    assert counts[("arabian_sea", "dev")] == 5 and counts[("equatorial", "dev")] == 3
    assert a["platform_number"].is_unique


def _matches(pred: np.ndarray, obs: np.ndarray) -> pd.DataFrame:
    n = obs.size
    return pd.DataFrame(
        {
            "platform_number": [f"p{k % 20}" for k in range(n)],
            "cycle": np.arange(n) // 20,
            "time": pd.Timestamp("2019-05-01"),
            "lat": 12.0,
            "lon": 88.0,
            "depth_m": np.where(np.arange(n) % 2 == 0, 92.326, 643.567),
            "temp_obs_c": obs,
            "temp_pred_c": pred,
        }
    )


def test_spread_skill_reports_a_calibrated_ensemble() -> None:
    rng = np.random.default_rng(1)
    n, sd = 4000, 0.5
    truth_center = rng.normal(25, 1, n)
    obs = truth_center + rng.normal(0, sd, n)  # error of the mean ~ sd
    members = [_matches(truth_center + rng.normal(0, sd * np.sqrt(3), n), obs) for _ in range(3)]
    table = ss.spread_skill(ss.combine(members)).set_index("group")
    assert table.loc["all", "n_obs"] == n
    assert 0.4 < table.loc["all", "frac_within_1sd_scaled"] < 0.8
    assert table.loc["bob_50_300", "n_obs"] == n // 2


def test_spread_skill_flags_a_spread_that_is_too_small() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    obs = rng.normal(25, 1, n)
    base = obs + 0.8  # shared bias every member has
    members = [_matches(base + rng.normal(0, 0.05, n), obs) for _ in range(3)]
    table = ss.spread_skill(ss.combine(members)).set_index("group")
    assert table.loc["all", "ratio"] > 5
    assert table.loc["all", "frac_within_2sd"] < 0.2
