import numpy as np
import pandas as pd

from retail_engine.module2_pooled import (
    _build_item_store_recommendations,
    _fit_department,
    _select_stratified_series,
)


def test_select_stratified_series_balances_departments_and_caps_size():
    eligible = pd.DataFrame(
        [
            {
                "dept_id": department,
                "item_id": f"{department}_{rank}",
                "store_id": "CA_1",
                "total_units": 100 - rank,
            }
            for department in ("A", "B", "C", "D")
            for rank in range(5)
        ]
    )

    selected = _select_stratified_series(eligible, max_series=10)

    assert len(selected) == 10
    assert selected.groupby("dept_id").size().to_dict() == {
        "A": 3,
        "B": 3,
        "C": 2,
        "D": 2,
    }
    assert selected.groupby("dept_id")["total_units"].max().eq(100).all()


def test_select_stratified_series_redistributes_quota_from_small_department():
    eligible = pd.DataFrame(
        [
            {
                "dept_id": department,
                "item_id": f"{department}_{rank}",
                "store_id": "CA_1",
                "total_units": 100 - rank,
            }
            for department, size in (("A", 5), ("B", 5), ("C", 5), ("D", 1))
            for rank in range(size)
        ]
    )

    selected = _select_stratified_series(eligible, max_series=10)

    assert len(selected) == 10
    assert selected.groupby("dept_id").size().to_dict() == {
        "A": 3,
        "B": 3,
        "C": 3,
        "D": 1,
    }


def test_fit_department_uses_clustered_two_way_fixed_effects():
    rng = np.random.default_rng(17)
    series_count = 24
    weeks = 20
    data = pd.DataFrame(
        {
            "item_id": np.repeat([f"item_{i}" for i in range(series_count)], weeks),
            "store_id": np.repeat(
                [f"store_{i % 4}" for i in range(series_count)], weeks
            ),
            "dept_id": "FOODS_1",
            "week_start": np.tile(
                pd.date_range("2015-01-03", periods=weeks, freq="W-SAT"),
                series_count,
            ),
        }
    )
    item_effect = np.repeat(rng.normal(size=series_count), weeks)
    week_index = np.tile(np.arange(weeks), series_count)
    price_signal = (
        np.repeat(rng.normal(scale=0.1, size=series_count), weeks)
        + np.sin(week_index * 0.43 + np.repeat(np.arange(series_count), weeks) * 0.31)
        * 0.2
    )
    store_week_effect = np.tile(np.cos(np.arange(weeks) * 0.27), series_count)
    log_units = (
        -0.7 * price_signal
        + item_effect
        + store_week_effect
        + rng.normal(scale=0.12, size=len(data))
    )
    data["price"] = np.exp(price_signal)
    data["units_sold"] = np.exp(log_units)

    result = _fit_department(data)

    assert result["dept_id"] == "FOODS_1"
    assert result["candidate_item_store_clusters"] == series_count
    assert result["feols_positive_sales_weekly_rows"] == series_count * weeks
    assert result["feols_elasticity"] < 0
    assert result["feols_standard_error"] > 0
    assert result["ppml_all_weekly_rows"] == series_count * weeks
    assert result["ppml_standard_error"] > 0


def test_ppml_recovers_known_price_elasticity_with_two_fixed_effects():
    rng = np.random.default_rng(123)
    series_count = 80
    weeks = 40
    data = pd.DataFrame(
        {
            "item_id": np.repeat([f"item_{i}" for i in range(series_count)], weeks),
            "store_id": "CA_1",
            "dept_id": "FOODS_1",
            "week_start": np.tile(
                pd.date_range("2015-01-03", periods=weeks, freq="W-SAT"),
                series_count,
            ),
        }
    )
    item_index = np.repeat(np.arange(series_count), weeks)
    week_index = np.tile(np.arange(weeks), series_count)
    log_price = rng.normal(scale=0.25, size=len(data))
    item_effect = rng.normal(loc=1.5, scale=0.2, size=series_count)[item_index]
    week_effect = 0.2 * np.sin(week_index / 5)
    true_elasticity = -1.5
    expected_units = np.exp(
        item_effect + week_effect + true_elasticity * log_price
    )
    data["price"] = np.exp(log_price)
    data["units_sold"] = rng.poisson(expected_units)

    result = _fit_department(data)

    assert abs(result["ppml_elasticity"] - true_elasticity) < 0.25
    assert result["ppml_all_weekly_rows"] == series_count * weeks


def test_per_series_dml_is_only_a_flag_not_the_default_action():
    selected = pd.DataFrame(
        [
            {
                "item_id": "FOODS_1_001",
                "store_id": "CA_1",
                "dept_id": "FOODS_1",
            }
        ]
    )
    pooled = pd.DataFrame(
        [
            {
                "dept_id": "FOODS_1",
                "ppml_elasticity": -1.4,
                "ppml_ci_lower": -1.8,
                "ppml_ci_upper": -1.1,
            }
        ]
    )
    dml = pd.DataFrame(
        [
            {
                "item_id": "FOODS_1_001",
                "store_id": "CA_1",
                "elasticity": 0.3,
                "demand_type": "Unclear",
            }
        ]
    )

    result = _build_item_store_recommendations(selected, pooled, dml).iloc[0]

    assert result["pooled_demand_type"] == "Price-sensitive"
    assert result["pricing_recommendation"] == "Test a controlled discount or promotion"
    assert result["dml_review_flag"] == "DML class differs from pooled default"


def test_pooled_action_requires_upper_interval_bound_below_minus_one():
    selected = pd.DataFrame(
        [
            {
                "item_id": "FOODS_2_001",
                "store_id": "CA_1",
                "dept_id": "FOODS_2",
            }
        ]
    )
    pooled = pd.DataFrame(
        [
            {
                "dept_id": "FOODS_2",
                "ppml_elasticity": -1.058,
                "ppml_ci_lower": -1.312,
                "ppml_ci_upper": -0.804,
            }
        ]
    )

    result = _build_item_store_recommendations(
        selected, pooled, pd.DataFrame()
    ).iloc[0]

    assert result["pooled_demand_type"] == "Less sensitive"
    assert result["pricing_recommendation"] == "Protect price or test a small increase"
