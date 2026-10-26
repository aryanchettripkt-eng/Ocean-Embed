"""Channel order and count (PRD_AI_ML §5.1). The order is defined only in channel_list."""

from __future__ import annotations

import pytest

from oceanscope_ml.config import ExperimentConfig
from oceanscope_ml.data.channels import channel_names


def test_default_channel_count_is_122(config: ExperimentConfig) -> None:
    # 11 dynamic vars x 7 history days, 2 long-history vars x 12 frames, 16 static, 5 extra.
    assert len(channel_names(config)) == 122


def test_default_channel_order(config: ExperimentConfig) -> None:
    names = channel_names(config)
    assert names[0] == "sla@t-6"  # oldest history day first
    assert names[76] == "precip@t-0"  # newest dynamic channel
    assert names[77] == "sla@mean56-60"  # oldest long-history frame first
    assert names[100] == "adt@mean1-5"
    assert names[101] == "bathymetry"
    assert names[102] == "woa@0.494m"
    assert names[116] == "woa@1062.44m"
    assert names[117:] == ["doy_sin", "doy_cos", "lat", "lon", "ocean_mask"]


def test_indices_add_channels_before_static(config: ExperimentConfig) -> None:
    config.inputs["indices_enabled"] = True
    names = channel_names(config)
    assert len(names) == 126
    assert names[101:105] == ["nino34", "dmi", "mjo_rmm1", "mjo_rmm2"]
    assert names[105] == "bathymetry"


def test_unknown_extra_channel_is_refused(config: ExperimentConfig) -> None:
    config.inputs["extra"] = ["doy_sin", "not_a_channel"]
    with pytest.raises(ValueError, match="Unknown extra inputs"):
        channel_names(config)
