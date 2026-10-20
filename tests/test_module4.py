import pandas as pd
import pytest

from retail_engine.module4 import (
    OUTPUT_COLUMNS,
    optimize_markdowns,
)


def _product(
    product_id: int,
    stock: int,
    historical_sales: int,
    sales_after: tuple[int, int, int, int],
) -> dict[str, object]:
    return {
        "Product_ID": product_id,
        "Category": "Skincare",
        "Brand": "BrandA",
        "Original_Price": 100.0,
        "Stock_Level": stock,
        "Historical_Sales": historical_sales,
        "Markdown_1": 0.1,
        "Markdown_2": 0.2,
        "Markdown_3": 0.3,
        "Markdown_4": 0.4,
        "Sales_After_M1": sales_after[0],
        "Sales_After_M2": sales_after[1],
        "Sales_After_M3": sales_after[2],
        "Sales_After_M4": sales_after[3],
    }


def test_optimizer_deduplicates_caps_sales_and_includes_leftover_salvage():
    row = _product(1, stock=100, historical_sales=10, sales_after=(20, 30, 40, 50))
    frame = pd.DataFrame([row, row])

    results, audit = optimize_markdowns(frame)

    assert list(results.columns) == OUTPUT_COLUMNS
    assert len(results) == 1
    assert audit["source_rows"] == 2
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["analysis_products"] == 1
    assert results.loc[0, "predicted_units_sold"] <= results.loc[0, "stock_units"]
    assert results.loc[0, "leftover_units"] == pytest.approx(
        results.loc[0, "stock_units"] - results.loc[0, "predicted_units_sold"]
    )
    assert results.loc[0, "unit_cost"] == pytest.approx(60.0)
    assert results.loc[0, "salvage_value_per_unit"] == pytest.approx(12.0)
    assert results.loc[0, "predicted_profit"] == pytest.approx(
        results.loc[0, "sales_revenue"]
        + results.loc[0, "salvage_proceeds"]
        - results.loc[0, "inventory_cost"]
    )


def test_recommendation_selects_profit_maximizing_markdown():
    row = _product(1, stock=100, historical_sales=10, sales_after=(20, 30, 40, 50))
    result, _ = optimize_markdowns(pd.DataFrame([row]))

    assert result.loc[0, "recommended_markdown"] == pytest.approx(0.10)
    assert result.loc[0, "predicted_units_sold"] == pytest.approx(100)
    assert result.loc[0, "recommended_price"] == pytest.approx(90.0)
    assert result.loc[0, "candidate_markdowns_evaluated"] == 6


def test_hand_worked_demand_curve_and_profit_maximum():
    row = _product(2, stock=200, historical_sales=10, sales_after=(12, 14, 16, 18))
    row.update(
        {
            "Markdown_1": 0.1,
            "Markdown_2": 0.2,
            "Markdown_3": 0.3,
            "Markdown_4": 0.4,
            "Optimal Discount": 0.4,
        }
    )

    result, _ = optimize_markdowns(pd.DataFrame([row]))
    recommendation = result.iloc[0]

    assert recommendation["demand_intercept_units_per_week"] == pytest.approx(10)
    assert recommendation["demand_slope_units_per_week_per_markdown"] == pytest.approx(
        20
    )
    assert recommendation["recommended_markdown"] == pytest.approx(0.20)
    assert recommendation["predicted_units_sold"] == pytest.approx(112)
    assert recommendation["leftover_units"] == pytest.approx(88)
    assert recommendation["sales_revenue"] == pytest.approx(8960)
    assert recommendation["inventory_cost"] == pytest.approx(12000)
    assert recommendation["salvage_proceeds"] == pytest.approx(1056)
    assert recommendation["predicted_profit"] == pytest.approx(-1984)
    assert recommendation["fixed_30_predicted_profit"] == pytest.approx(-2176)
    assert recommendation["profit_gain_vs_fixed_30"] == pytest.approx(192)
    assert recommendation["optimal_discount_reference"] == pytest.approx(0.4)
    assert recommendation["recommendation_status"].startswith(
        "Scenario under assumptions"
    )
    alternate_reference = pd.DataFrame([{**row, "Optimal Discount": 0.2}])
    alternate_result, _ = optimize_markdowns(alternate_reference)
    assert alternate_result.loc[0, "recommended_markdown"] == pytest.approx(
        recommendation["recommended_markdown"]
    )
    assert alternate_result.loc[0, "predicted_profit"] == pytest.approx(
        recommendation["predicted_profit"]
    )
    assert alternate_result.loc[0, "optimal_discount_reference"] == pytest.approx(0.2)


def test_economic_sensitivity_reports_all_nine_assumption_pairs():
    from retail_engine.module4 import markdown_economic_sensitivity

    row = _product(1, stock=100, historical_sales=10, sales_after=(20, 30, 40, 50))
    sensitivity = markdown_economic_sensitivity(pd.DataFrame([row]))

    assert len(sensitivity) == 9
    assert set(sensitivity["unit_cost_rate"]) == {0.5, 0.6, 0.7}
    assert set(sensitivity["salvage_rate_of_cost"]) == {0.1, 0.2, 0.3}
    assert sensitivity["scenario_label"].str.startswith(
        "Scenario under assumptions"
    ).all()


def test_weeks_cover_rule_holds_product_if_baseline_covers_horizon():
    row = _product(1, stock=80, historical_sales=10, sales_after=(20, 30, 40, 50))
    result, _ = optimize_markdowns(
        pd.DataFrame([row]),
        horizon_weeks=8,
    )

    assert result.loc[0, "weeks_cover"] == pytest.approx(8)
    assert not result.loc[0, "markdown_eligible_by_weeks_cover"]
    assert result.loc[0, "recommended_markdown"] == pytest.approx(0)


def test_historical_sales_period_scales_weekly_demand_and_stock_cover():
    row = _product(1, stock=100, historical_sales=130, sales_after=(260, 390, 520, 650))
    result, _ = optimize_markdowns(
        pd.DataFrame([row]),
        historical_sales_period_weeks=13,
        horizon_weeks=8,
    )

    assert result.loc[0, "demand_intercept_units_per_week"] == pytest.approx(10)
    assert result.loc[0, "weeks_cover"] == pytest.approx(10)
    assert result.loc[0, "markdown_eligible_by_weeks_cover"]


def test_zero_stock_has_no_sales_and_zero_profit_for_all_candidates():
    row = _product(1, stock=0, historical_sales=10, sales_after=(20, 30, 40, 50))
    result, _ = optimize_markdowns(pd.DataFrame([row]))

    assert result.loc[0, "recommended_markdown"] == 0.0
    assert result.loc[0, "predicted_units_sold"] == 0
    assert result.loc[0, "leftover_units"] == 0
    assert result.loc[0, "predicted_profit"] == 0


def test_leftover_inventory_receives_salvage_credit():
    row = _product(1, stock=100, historical_sales=0, sales_after=(0, 0, 0, 0))
    result, _ = optimize_markdowns(pd.DataFrame([row]))

    assert result.loc[0, "predicted_units_sold"] == 0
    assert result.loc[0, "leftover_units"] == 100
    assert result.loc[0, "salvage_proceeds"] == pytest.approx(1200.0)
    assert result.loc[0, "inventory_cost"] == pytest.approx(6000.0)
    assert result.loc[0, "predicted_profit"] == pytest.approx(-4800.0)


def test_conflicting_product_ids_are_rejected():
    first = _product(1, stock=100, historical_sales=10, sales_after=(20, 30, 40, 50))
    second = {**first, "Stock_Level": 120}

    with pytest.raises(ValueError, match="Conflicting rows"):
        optimize_markdowns(pd.DataFrame([first, second]))
