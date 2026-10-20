from pathlib import Path

import pandas as pd
import pytest

from retail_engine.module3_causalml import _prepare_panel, _weekly_panel


def test_panel_features_use_only_prior_weeks():
    panel = pd.DataFrame(
        {
            "item_id": ["item"] * 7,
            "store_id": ["CA_1"] * 7,
            "cat_id": ["FOODS"] * 7,
            "week_start": pd.date_range("2015-01-03", periods=7, freq="7D"),
            "units_sold": [2, 4, 6, 8, 10, 12, 14],
            "snap_share": [0.0] * 7,
            "event_week": [0.0] * 7,
            "treatment": [0, 0, 0, 0, 1, 0, 1],
            "lag_1_units": [None, 2, 4, 6, 8, 10, 12],
            "lag_4_units": [None, None, None, None, 2, 4, 6],
            "prior_4_week_mean": [None, None, None, None, 5, 7, 9],
        }
    )

    prepared, features = _prepare_panel(panel)

    assert prepared.iloc[0]["week_start"] == pd.Timestamp("2015-01-31")
    assert prepared.iloc[0]["treatment"] == 1
    assert prepared.iloc[0]["lag_1_units"] == 8
    assert prepared.iloc[0]["lag_4_units"] == 2
    assert prepared.iloc[0]["prior_4_week_mean"] == 5
    assert {"cat_id_FOODS", "store_id_CA_1"} <= set(features.columns)
    assert features.isna().sum().sum() == 0


def test_weekly_panel_rejects_nonpositive_series_limit(tmp_path):
    with pytest.raises(ValueError, match="max_series must be positive"):
        _weekly_panel(tmp_path / "missing.parquet", 0)


def test_weekly_panel_reports_missing_input():
    with pytest.raises(FileNotFoundError, match="Module 1 Parquet"):
        _weekly_panel(Path("missing.parquet"), 1)
