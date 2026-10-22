import numpy as np
import pandas as pd
import pytest

from retail_engine.module2_daily import (
    RESULT_COLUMNS,
    WEEKDAYS,
    compare_with_weekly,
    comparison_markdown,
    daily_control_matrix,
    estimate_daily_elasticities,
)


def _daily_series(item_id="A", store_id="CA_1", elasticity=-1.5, seed=0, weeks=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-03", periods=weeks * 7, freq="D")
    weekly_log_price = rng.normal(0, 0.25, weeks)
    log_price = np.repeat(weekly_log_price, 7)
    weekday_effect = np.array([0.0, 0.1, 0.1, 0.2, 0.4, 0.8, 0.7])[dates.dayofweek]
    log_units = 3.0 + elasticity * log_price + weekday_effect + rng.normal(0, 0.05, len(dates))
    return pd.DataFrame(
        {
            "item_id": item_id,
            "store_id": store_id,
            "date": dates,
            "units_sold": np.expm1(log_units),
            "price": np.exp(log_price),
            "snap": 0.0,
            "promo": 0.0,
            "event": 0.0,
            "weekday": [WEEKDAYS[d] for d in dates.dayofweek],
            "month": dates.month,
        }
    )


def test_control_matrix_has_flags_weekday_and_month_indicators():
    daily = _daily_series(weeks=10)
    controls = daily_control_matrix(daily)

    assert controls.shape == (len(daily), 3 + 6 + 11)
    assert set(np.unique(controls)) <= {0.0, 1.0}
    # Monday is the baseline weekday, so a Tuesday row has exactly the first weekday dummy on.
    tuesday = daily["weekday"].eq("Tuesday").to_numpy()
    assert (controls[tuesday, 3] == 1.0).all()
    assert controls[~tuesday, 3].max() == 0.0


def test_daily_estimate_recovers_known_elasticity_despite_weekday_effects():
    daily = _daily_series(elasticity=-1.5)

    result = estimate_daily_elasticities(daily, folds=3, trees=30).iloc[0]

    assert result["fit_status"] == "estimated"
    assert result["daily_elasticity"] == pytest.approx(-1.5, abs=0.5)
    assert result["daily_ci_lower"] < result["daily_elasticity"] < result["daily_ci_upper"]
    assert result["days_used"] == 60 * 7


def test_daily_results_compare_with_weekly_and_summarise():
    daily = pd.concat(
        [_daily_series("A", "CA_1", -1.5, seed=1), _daily_series("B", "TX_1", -0.5, seed=2)]
    )
    results = estimate_daily_elasticities(daily, folds=3, trees=30)
    assert list(results.columns) == RESULT_COLUMNS
    weekly = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "store_id": ["CA_1", "TX_1"],
            "elasticity": [-1.2, -0.4],
            "ci_lower": [-2.0, -1.0],
            "ci_upper": [-0.4, 0.2],
            "demand_type": ["Less sensitive", "Unclear"],
        }
    )

    compared = compare_with_weekly(results, weekly)
    text = comparison_markdown(compared)

    assert compared["weekly_elasticity"].tolist() == [-1.2, -0.4]
    assert "Series estimated: 2 of 2" in text
    assert "Same sign as the weekly estimate: 100%" in text


def test_series_with_too_few_days_is_reported_not_estimable():
    result = estimate_daily_elasticities(_daily_series(weeks=2), folds=5, trees=10).iloc[0]

    assert result["fit_status"].startswith("not_estimable")
    assert result["daily_demand_type"] == "Unclear"
