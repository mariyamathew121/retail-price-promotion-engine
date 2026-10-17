import numpy as np
import pandas as pd

from retail_engine.module2_pooled import (
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
    assert result["positive_sales_weekly_rows"] == series_count * weeks
    assert result["elasticity"] < 0
    assert result["standard_error"] > 0
