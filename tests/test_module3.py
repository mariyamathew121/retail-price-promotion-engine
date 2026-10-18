import pandas as pd
import pytest

from retail_engine.module3 import _evaluate_series_promotions


def _promotion_example(
    item_id: str,
    promo_units: tuple[int, int],
    promo_price: float,
) -> pd.DataFrame:
    week_starts = pd.date_range("2015-01-03", periods=8, freq="7D")
    units = [10, 10, 10, 10, *promo_units, 8, 8]
    prices = [10, 10, 10, 10, promo_price, promo_price, 10, 10]
    promo_flags = [0, 0, 0, 0, 1, 1, 0, 0]
    return pd.DataFrame(
        {
            "item_id": item_id,
            "store_id": "CA_1",
            "cat_id": "FOODS",
            "week_start": week_starts,
            "units_sold": units,
            "price": prices,
            "regular_price": 10.0,
            "promo_flag": promo_flags,
            "discount_depth": 0.1,
        }
    )


def test_three_hand_calculated_promotion_examples():
    cases = [
        _promotion_example("continue", (20, 20), 9.0),
        _promotion_example("reduce", (12, 12), 8.0),
        _promotion_example("stop", (8, 8), 9.0),
    ]
    results = {
        result["item_id"]: result
        for case in cases
        for result in _evaluate_series_promotions(case, gross_margin_rate=0.30)
    }

    assert results["continue"]["baseline_sales"] == 20
    assert results["continue"]["promo_sales"] == 40
    assert results["continue"]["incremental_units"] == 20
    assert results["continue"]["baseline_revenue"] == 200
    assert results["continue"]["promo_revenue"] == 360
    assert results["continue"]["incremental_revenue"] == 160
    assert results["continue"]["baseline_profit"] == 60
    assert results["continue"]["promo_profit"] == 80
    assert results["continue"]["incremental_margin"] == 20
    assert results["continue"]["discount_cost"] == 40
    assert results["continue"]["margin_rate_used"] == 0.30
    assert results["continue"]["promotion_ROI"] == 0.5
    assert results["continue"]["recommendation"] == "Continue promotion"

    assert results["reduce"]["incremental_units"] == 4
    assert results["reduce"]["baseline_revenue"] == 200
    assert results["reduce"]["promo_revenue"] == 192
    assert results["reduce"]["incremental_revenue"] == -8
    assert results["reduce"]["baseline_profit"] == 60
    assert results["reduce"]["promo_profit"] == 24
    assert results["reduce"]["incremental_margin"] == -36
    assert results["reduce"]["discount_cost"] == 48
    assert results["reduce"]["promotion_ROI"] == -0.75
    assert results["reduce"]["recommendation"] == "Reduce discount"

    assert results["stop"]["incremental_units"] == -4
    assert results["stop"]["baseline_revenue"] == 200
    assert results["stop"]["promo_revenue"] == 144
    assert results["stop"]["incremental_revenue"] == -56
    assert results["stop"]["baseline_profit"] == 60
    assert results["stop"]["promo_profit"] == 32
    assert results["stop"]["incremental_margin"] == -28
    assert results["stop"]["discount_cost"] == 16
    assert results["stop"]["promotion_ROI"] == -1.75
    assert results["stop"]["recommendation"] == "Stop promotion"

    for result in results.values():
        assert result["post_promo_weeks_observed"] == 2
        assert result["post_promo_dip_units"] == -4


def test_category_margin_is_used_without_an_override():
    result = _evaluate_series_promotions(
        _promotion_example("foods_default", (20, 20), 9.0)
    )[0]

    assert result["cat_id"] == "FOODS"
    assert result["avg_discount_depth"] == 0.1
    assert result["margin_rate_used"] == 0.25
    assert result["baseline_profit"] == 50
    assert result["promo_profit"] == 60
    assert result["incremental_margin"] == 10


def test_persistent_low_price_is_flagged_for_review():
    weekly = _promotion_example("persistent_price", (20, 20), 9.0)
    weekly.loc[6, "price"] = 9.0

    result = _evaluate_series_promotions(weekly, gross_margin_rate=0.30)[0]

    assert result["price_persists"] is True
    assert result["recommendation"] == "Possible permanent price change - review"


def test_persistent_price_review_takes_precedence_when_baseline_is_insufficient():
    weekly = _promotion_example("persistent_short_history", (20, 20), 9.0)
    weekly = weekly.iloc[3:].copy()
    weekly.loc[6, "price"] = 9.0

    result = _evaluate_series_promotions(weekly)[0]

    assert result["analysis_status"] == "insufficient_baseline"
    assert result["price_persists"] is True
    assert result["recommendation"] == "Possible permanent price change - review"


def test_insufficient_history_is_reported_without_a_recommendation():
    weekly = _promotion_example("short_history", (20, 20), 9.0).iloc[3:].copy()

    result = _evaluate_series_promotions(weekly)[0]

    assert result["analysis_status"] == "insufficient_baseline"
    assert result["baseline_weeks_used"] == 1
    assert result["baseline_sales"] is None
    assert result["recommendation"] == "Review - insufficient baseline"


@pytest.mark.parametrize("margin_rate", [0.0, 1.0, -0.1, 1.1])
def test_invalid_margin_rates_are_rejected(margin_rate):
    with pytest.raises(ValueError, match="gross_margin_rate"):
        _evaluate_series_promotions(
            _promotion_example("invalid_margin", (20, 20), 9.0),
            gross_margin_rate=margin_rate,
        )
