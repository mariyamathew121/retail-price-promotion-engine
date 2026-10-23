import numpy as np
import pandas as pd
import pytest

from retail_engine.forecast_holdout import forecast_series, wape


def _series(weeks=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "item_id": "A",
            "store_id": "CA_1",
            "week_start": pd.date_range("2015-01-03", periods=weeks, freq="7D"),
            "units_sold": rng.poisson(20, weeks).astype(float),
            "price": 2.0 + rng.normal(0, 0.05, weeks),
            "snap_share": 0.0,
            "event_week": 0.0,
            "promo_week": 0.0,
        }
    )


def test_holdout_predicts_only_the_last_weeks_and_never_uses_them_to_fit():
    weekly = _series()
    out = forecast_series(weekly, holdout_weeks=13)

    assert len(out) == 13
    assert out["week_start"].min() == weekly["week_start"].iloc[-13]
    # Changing only the held-out weeks must not change the training-period forecast.
    changed = weekly.copy()
    changed.loc[changed.index[-13:], "units_sold"] = 500.0
    out_changed = forecast_series(changed, holdout_weeks=13)
    assert out_changed["naive_forecast"].iloc[0] == pytest.approx(out["naive_forecast"].iloc[0])


def test_wape_is_error_over_total_units():
    assert wape(np.array([10.0, 10.0]), np.array([8.0, 14.0])) == pytest.approx(0.3)
    assert np.isnan(wape(np.array([0.0]), np.array([1.0])))


def test_short_series_are_skipped():
    assert forecast_series(_series(weeks=15), holdout_weeks=13).empty


def test_pooled_holdout_scores_every_eligible_series_on_the_held_out_weeks():
    from retail_engine.forecast_holdout import run_pooled_holdout

    weekly = pd.concat(
        [_series(60).assign(item_id="A"), _series(60).assign(item_id="B", store_id="TX_1")]
    )
    predictions = run_pooled_holdout(weekly, holdout_weeks=13)

    assert len(predictions) == 2 * 13
    assert set(predictions["pooled_forecast"].ge(0)) == {True}
