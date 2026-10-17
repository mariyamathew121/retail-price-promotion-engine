import json

import pandas as pd
import pytest

from retail_engine.module1 import (
    OUTPUT_COLUMNS,
    _add_rolling_regular_prices,
    build_clean_retail_data,
)


@pytest.fixture
def m5_inputs(tmp_path):
    sales_path = tmp_path / "sales_train_evaluation.csv"
    prices_path = tmp_path / "sell_prices.csv"
    calendar_path = tmp_path / "calendar.csv"

    pd.DataFrame(
        [
            {
                "id": "ITEM_1_CA_1_evaluation",
                "item_id": "ITEM_1",
                "dept_id": "DEPT_1",
                "cat_id": "CAT_1",
                "store_id": "CA_1",
                "state_id": "CA",
                "d_1": 4,
                "d_2": 6,
                "d_3": 2,
            },
            {
                "id": "ITEM_2_CA_1_evaluation",
                "item_id": "ITEM_2",
                "dept_id": "DEPT_1",
                "cat_id": "CAT_1",
                "store_id": "CA_1",
                "state_id": "CA",
                "d_1": 0,
                "d_2": 1,
                "d_3": 0,
            },
        ]
    ).to_csv(sales_path, index=False)
    pd.DataFrame(
        [
            {"store_id": "CA_1", "item_id": "ITEM_1", "wm_yr_wk": 1, "sell_price": 10.0},
            {"store_id": "CA_1", "item_id": "ITEM_1", "wm_yr_wk": 2, "sell_price": 8.0},
            {"store_id": "CA_1", "item_id": "ITEM_1", "wm_yr_wk": 3, "sell_price": 10.0},
            {"store_id": "CA_1", "item_id": "ITEM_2", "wm_yr_wk": 1, "sell_price": 5.0},
            {"store_id": "CA_1", "item_id": "ITEM_2", "wm_yr_wk": 3, "sell_price": 5.0},
        ]
    ).to_csv(prices_path, index=False)
    pd.DataFrame(
        [
            {
                "d": "d_1",
                "date": "2011-01-29",
                "wm_yr_wk": 1,
                "weekday": "Saturday",
                "month": 1,
                "event_name_1": None,
                "event_type_1": None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": 1,
                "snap_TX": 0,
                "snap_WI": 0,
            },
            {
                "d": "d_2",
                "date": "2011-01-30",
                "wm_yr_wk": 2,
                "weekday": "Sunday",
                "month": 1,
                "event_name_1": "Event A",
                "event_type_1": "Cultural",
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": 0,
                "snap_TX": 1,
                "snap_WI": 0,
            },
            {
                "d": "d_3",
                "date": "2011-01-31",
                "wm_yr_wk": 3,
                "weekday": "Monday",
                "month": 1,
                "event_name_1": None,
                "event_type_1": None,
                "event_name_2": "Event B",
                "event_type_2": "Sporting",
                "snap_CA": 1,
                "snap_TX": 0,
                "snap_WI": 0,
            },
        ]
    ).to_csv(calendar_path, index=False)
    return sales_path, prices_path, calendar_path


def test_builds_expected_output_and_audit(m5_inputs, tmp_path):
    sales, prices, calendar = m5_inputs
    output = tmp_path / "clean_retail_data.parquet"

    audit = build_clean_retail_data(sales, prices, calendar, output, chunk_skus=1)
    result = pd.read_parquet(output)

    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert len(result) == 5
    assert result["price"].notna().all()
    assert audit["daily_rows_before_price_filter"] == 6
    assert audit["dropped_rows_without_weekly_price"] == 1
    assert audit["active_rows_missing_price_after_filter"] == 0
    assert audit["spot_check_sku_store_combinations"] == 2
    assert audit["flagged_rows_with_discount_below_threshold"] == 0
    assert audit["promo_share"] == pytest.approx(1 / 5)
    assert (tmp_path / "clean_retail_data_promo_spot_checks.png").is_file()

    item_one = result.loc[result["item_id"] == "ITEM_1"].sort_values("date")
    assert item_one["regular_price"].tolist() == [10.0, 10.0, 10.0]
    assert item_one["promo_flag"].tolist() == [0, 1, 0]
    assert item_one["discount_depth"].tolist() == pytest.approx([0.0, 0.2, 0.0])
    assert item_one["snap_flag"].tolist() == [1, 0, 1]
    assert item_one["event_name"].tolist()[1:] == ["Event A", "Event B"]

    audit_file = output.with_name("clean_retail_data_audit.json")
    saved_audit = json.loads(audit_file.read_text(encoding="utf-8"))
    assert saved_audit["promo_flag_rows"] == 1


def test_rolling_max_excludes_current_and_uses_prior_eight_calendar_weeks():
    weeks = pd.date_range("2020-01-04", periods=10, freq="7D")
    prices = pd.DataFrame(
        {
            "store_id": ["STORE"] * 10,
            "item_id": ["ITEM"] * 10,
            "wm_yr_wk": range(10),
            "week_start": weeks,
            "sell_price": [10.0, 9.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.0],
        }
    )

    result = _add_rolling_regular_prices(prices)

    assert result.loc[0, "regular_price"] == 10.0
    assert not bool(result.loc[0, "has_price_history"])
    assert result.loc[0, "promo_flag"] == 0
    assert result.loc[1, "regular_price"] == 10.0
    assert result.loc[1, "discount_depth"] == pytest.approx(0.05)
    assert result.loc[1, "promo_flag"] == 0
    assert result.loc[8, "regular_price"] == 10.0
    assert result.loc[9, "regular_price"] == 9.5
    assert result.loc[9, "discount_depth"] == pytest.approx(1 - 8 / 9.5)
    assert result.loc[9, "promo_flag"] == 1

    current_week_excluded = _add_rolling_regular_prices(
        pd.DataFrame(
            {
                "store_id": ["STORE", "STORE"],
                "item_id": ["OTHER_ITEM", "OTHER_ITEM"],
                "wm_yr_wk": [1, 2],
                "week_start": weeks[:2],
                "sell_price": [10.0, 12.0],
            }
        )
    )
    assert current_week_excluded.loc[1, "regular_price"] == 10.0
    assert current_week_excluded.loc[1, "promo_flag"] == 0


def test_fails_when_calendar_day_is_missing(m5_inputs, tmp_path):
    sales, prices, calendar = m5_inputs
    calendar_data = pd.read_csv(calendar).iloc[:2]
    calendar_data.to_csv(calendar, index=False)

    with pytest.raises(ValueError, match="missing 1 sales day key"):
        build_clean_retail_data(sales, prices, calendar, tmp_path / "out.parquet")
