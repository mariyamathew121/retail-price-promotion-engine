import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retail_engine.module2 import (
    _crossfit_dml,
    _hac_standard_error,
    _weekly_control_matrix,
    estimate_price_elasticity,
)


def test_crossfit_dml_recovers_simulated_elasticity():
    rng = np.random.default_rng(31)
    n = 3_000
    controls = rng.uniform(-1, 1, size=(n, 3))
    log_price = (
        0.9 * controls[:, 0]
        + controls[:, 1] ** 2
        + 0.5 * np.sin(np.pi * controls[:, 2])
        + rng.normal(scale=0.6, size=n)
    )
    true_elasticity = -1.5
    log_units = (
        true_elasticity * log_price
        + 1.5 * controls[:, 0]
        + np.cos(np.pi * controls[:, 1])
        + rng.normal(scale=0.3, size=n)
    )

    estimate, standard_error = _crossfit_dml(
        log_price,
        log_units,
        controls,
        np.arange(n),
        folds=5,
        trees=80,
        seed=31,
    )

    assert estimate == pytest.approx(true_elasticity, abs=0.2)
    assert standard_error > 0


def test_estimates_eligible_series_and_writes_outputs(tmp_path):
    input_path = tmp_path / "clean_retail_data.parquet"
    weeks = pd.date_range("2020-01-04", periods=64, freq="7D")
    records = []
    for index, week in enumerate(weeks):
        price = [10.0, 9.0, 8.0][(index // 12) % 3]
        quantity = (
            0 if index % 17 == 0 else max(1, round(18 - 0.8 * price + (index % 3)))
        )
        for day in range(7):
            records.append(
                {
                    "item_id": "TEST_ITEM",
                    "store_id": "TEST_STORE",
                    "date": week + pd.Timedelta(days=day),
                    "units_sold": quantity,
                    "price": price,
                    "snap_flag": int(index % 4 == 0),
                    "event_name": "Test Event" if index % 20 == 0 else None,
                    "promo_flag": int(index % 12 == 0),
                }
            )
    table = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
    pq.write_table(table, input_path)
    output_path = tmp_path / "price_elasticities.csv"

    audit = estimate_price_elasticity(
        input_path,
        output_path,
        top_n=10,
        min_weeks=52,
        min_positive_weeks=40,
        min_price_levels=3,
        min_price_range=0.05,
        folds=4,
        trees=10,
    )

    result = pd.read_csv(output_path)
    assert audit["selected_combinations"] == 1
    assert audit["estimated_combinations"] == 1
    assert result.loc[0, "item_id"] == "TEST_ITEM"
    assert result.loc[0, "fit_status"] == "estimated"
    assert np.isfinite(result.loc[0, "elasticity"])
    assert result.loc[0, "conf_interval"].startswith("[")
    assert np.isfinite(result.loc[0, "naive_elasticity"])
    assert result.loc[0, "naive_conf_interval"].startswith("[")
    assert result.loc[0, "weeks_used"] == 64
    assert {"naive_elasticity", "ols_elasticity", "elasticity"}.issubset(
        result.columns
    )
    assert (tmp_path / "price_elasticities_selected_combinations.csv").is_file()
    assert (tmp_path / "price_elasticities_audit.json").is_file()
    assert (tmp_path / "price_elasticities.md").is_file()
    assert json.loads(
        (tmp_path / "price_elasticities_audit.json").read_text(encoding="utf-8")
    )["estimated_combinations"] == 1


def test_control_matrix_has_promo_and_seasonality_without_time_trend():
    weekly = pd.DataFrame(
        {
            "week_start": pd.date_range("2020-01-04", periods=3, freq="7D"),
            "snap_share": [0.0, 0.5, 1.0],
            "event_week": [0.0, 1.0, 0.0],
            "promo_week": [0.0, 1.0, 1.0],
        }
    )

    controls = _weekly_control_matrix(weekly)

    assert controls.shape == (3, 5)
    np.testing.assert_array_equal(controls[:, 2], weekly["snap_share"])
    np.testing.assert_array_equal(controls[:, 3], weekly["event_week"])
    np.testing.assert_array_equal(controls[:, 4], weekly["promo_week"])


def test_hac_matches_only_observations_at_exact_week_lags():
    influence = np.array([1.0, 2.0, -1.0])
    week_indices = np.array([0, 2, 3])
    n = len(influence)
    variance = float(np.dot(influence, influence) / n)
    # Exact pairs are (2, 3), (0, 2), and (0, 3) at lags 1, 2, and 3.
    variance += 2 * (1 - 1 / 5) * (2 * -1) / n
    variance += 2 * (1 - 2 / 5) * (1 * 2) / n
    variance += 2 * (1 - 3 / 5) * (1 * -1) / n
    expected = np.sqrt(max(variance, 0) / n)

    assert _hac_standard_error(influence, week_indices, max_lag=4) == pytest.approx(
        expected
    )
