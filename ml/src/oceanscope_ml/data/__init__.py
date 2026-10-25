"""Data loading, downloading, inspection, reduction, climatology, and preprocessing modules."""

from oceanscope_ml.data.climatology import (
    apply_climatology,
    evaluate_harmonic_climatology,
    fit_harmonic_climatology,
)
from oceanscope_ml.data.reduce import TargetGrid, area_mean_regrid_2d
from oceanscope_ml.data.transform import (
    TestLeakageError,
    check_no_test_leakage,
    compute_norm_stats,
    denormalize_inputs,
    denormalize_target,
    normalize_inputs,
    normalize_target,
)

__all__ = [
    "TargetGrid",
    "area_mean_regrid_2d",
    "fit_harmonic_climatology",
    "evaluate_harmonic_climatology",
    "apply_climatology",
    "TestLeakageError",
    "check_no_test_leakage",
    "compute_norm_stats",
    "normalize_inputs",
    "denormalize_inputs",
    "normalize_target",
    "denormalize_target",
]
