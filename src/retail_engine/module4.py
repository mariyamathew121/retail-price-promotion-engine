"""Simulate product-level markdown profit using the D2 dataset and labelled assumptions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HORIZON_WEEKS = 8
UNIT_COST_RATE = 0.60
SALVAGE_RATE_OF_COST = 0.20
MAX_MARKDOWN = 0.50
MARKDOWN_STEP = 0.10
SENSITIVITY_UNIT_COST_RATES = (0.50, 0.60, 0.70)
SENSITIVITY_SALVAGE_RATES = (0.10, 0.20, 0.30)
PERIOD_HORIZON_VARIANTS = (1, 13, 26)
HORIZON_VARIANTS = (4, 8, 16)
SEASON_ENDING_WEEKS = 4
WEEKS_COVER_THRESHOLD_RULE = (
    "A product is markdown-eligible only when opening-stock weeks cover "
    "strictly exceed the simulated demand horizon."
)
ACTION_NO_MARKDOWN = "No markdown - low stock"
ACTION_HOLD = "Hold price - markdown not profitable"
ACTION_DEEPER = "Deeper markdown - season ending, stock high"
ACTION_MARK_DOWN = "Mark down - high stock, weak sales"
ACTION_RULES = (
    f"{ACTION_NO_MARKDOWN}: weeks cover does not exceed the horizon. "
    f"{ACTION_HOLD}: eligible, but 0% has the highest expected margin. "
    f"{ACTION_DEEPER}: eligible, a markdown is selected, and the remaining "
    "horizon is at most the season-ending threshold. "
    f"{ACTION_MARK_DOWN}: eligible and a markdown is selected with more "
    "weeks remaining. The optimizer sets the depth; actions label which "
    "business rule applies."
)
MARKDOWN_COLUMNS = [f"Markdown_{stage}" for stage in range(1, 5)]
SALES_COLUMNS = [f"Sales_After_M{stage}" for stage in range(1, 5)]
INPUT_COLUMNS = [
    "Product_ID",
    "Category",
    "Brand",
    "Original_Price",
    "Stock_Level",
    "Historical_Sales",
    *MARKDOWN_COLUMNS,
    *SALES_COLUMNS,
]
OUTPUT_COLUMNS = [
    "SKU",
    "Category",
    "Brand",
    "original_price",
    "unit_cost",
    "salvage_value_per_unit",
    "current_stock",
    "horizon_weeks",
    "historical_sales_period_weeks",
    "weeks_cover",
    "markdown_eligible_by_weeks_cover",
    "at_risk_unsold",
    "demand_intercept_units_per_week",
    "demand_slope_units_per_week_per_markdown",
    "demand_fit_r_squared",
    "recommended_markdown",
    "recommended_price",
    "predicted_units_per_week",
    "uncapped_horizon_demand",
    "expected_sales",
    "leftover_inventory",
    "sales_revenue",
    "inventory_cost",
    "salvage_proceeds",
    "expected_margin",
    "fixed_30_markdown",
    "fixed_30_expected_margin",
    "profit_gain_vs_fixed_30",
    "optimal_discount_reference",
    "recommended_minus_optimal_discount",
    "action",
    "recommendation_status",
    "candidate_markdowns_evaluated",
]


def _validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(set(INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Markdown input is missing required columns: {missing}")
    numeric_columns = [
        "Original_Price",
        "Stock_Level",
        "Historical_Sales",
        *MARKDOWN_COLUMNS,
        *SALES_COLUMNS,
    ]
    if frame[numeric_columns].isna().any().any():
        raise ValueError("Markdown input contains missing required numeric values")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Markdown input contains non-finite numeric values")
    if frame["Original_Price"].le(0).any():
        raise ValueError("Original_Price must be positive")
    if frame["Stock_Level"].lt(0).any():
        raise ValueError("Stock_Level cannot be negative")
    if frame["Historical_Sales"].lt(0).any() or frame[SALES_COLUMNS].lt(0).any().any():
        raise ValueError("Sales values cannot be negative")
    if frame[MARKDOWN_COLUMNS].lt(0).any().any() or frame[MARKDOWN_COLUMNS].gt(
        1
    ).any().any():
        raise ValueError("Observed markdowns must be fractions between 0 and 1")


def _fit_demand_curve(
    product: pd.Series, historical_sales_period_weeks: int = 1
) -> tuple[float, float, float]:
    observed_markdowns = product[MARKDOWN_COLUMNS].to_numpy(dtype=float)
    observed_sales = (
        product[SALES_COLUMNS].to_numpy(dtype=float) / historical_sales_period_weeks
    )
    markdowns = np.concatenate(([0.0], observed_markdowns))
    weekly_demand = np.concatenate(
        (
            [
                float(product["Historical_Sales"])
                / historical_sales_period_weeks
            ],
            observed_sales,
        )
    )
    design = np.column_stack((np.ones(len(markdowns)), markdowns))
    intercept, slope = np.linalg.lstsq(design, weekly_demand, rcond=None)[0]
    fitted = design @ np.array([intercept, slope])
    residual_sum_squares = float(np.square(weekly_demand - fitted).sum())
    total_sum_squares = float(
        np.square(weekly_demand - weekly_demand.mean()).sum()
    )
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else 1.0
    )
    return float(intercept), float(slope), float(r_squared)


def _profit_curve(
    product: pd.Series,
    markdown_grid: np.ndarray,
    horizon_weeks: int,
    unit_cost_rate: float,
    salvage_rate_of_cost: float,
    historical_sales_period_weeks: int = 1,
) -> pd.DataFrame:
    intercept, slope, r_squared = _fit_demand_curve(
        product, historical_sales_period_weeks
    )
    original_price = float(product["Original_Price"])
    unit_cost = original_price * unit_cost_rate
    salvage_value_per_unit = unit_cost * salvage_rate_of_cost
    stock_units = float(product["Stock_Level"])
    weekly_demand = np.maximum(0.0, intercept + slope * markdown_grid)
    uncapped_horizon_demand = weekly_demand * horizon_weeks
    units_sold = np.minimum(uncapped_horizon_demand, stock_units)
    leftover_units = stock_units - units_sold
    prices = original_price * (1 - markdown_grid)
    sales_revenue = prices * units_sold
    inventory_cost = np.full_like(markdown_grid, stock_units * unit_cost)
    salvage_proceeds = leftover_units * salvage_value_per_unit
    predicted_profit = sales_revenue + salvage_proceeds - inventory_cost
    return pd.DataFrame(
        {
            "markdown": markdown_grid,
            "price": prices,
            "predicted_units_per_week": weekly_demand,
            "uncapped_horizon_demand": uncapped_horizon_demand,
            "predicted_units_sold": units_sold,
            "leftover_units": leftover_units,
            "sales_revenue": sales_revenue,
            "inventory_cost": inventory_cost,
            "salvage_proceeds": salvage_proceeds,
            "predicted_profit": predicted_profit,
            "demand_intercept_units_per_week": intercept,
            "demand_slope_units_per_week_per_markdown": slope,
            "demand_fit_r_squared": r_squared,
            "unit_cost": unit_cost,
            "salvage_value_per_unit": salvage_value_per_unit,
        }
    )


def _simulate_product(
    product: pd.Series,
    markdown_grid: np.ndarray,
    horizon_weeks: int,
    unit_cost_rate: float,
    salvage_rate_of_cost: float,
    optimal_discount_reference: float | None = None,
    historical_sales_period_weeks: int = 1,
    season_ending_weeks: int = SEASON_ENDING_WEEKS,
) -> dict[str, object]:
    curve = _profit_curve(
        product,
        markdown_grid,
        horizon_weeks,
        unit_cost_rate,
        salvage_rate_of_cost,
        historical_sales_period_weeks,
    )
    baseline_weekly_sales = (
        float(product["Historical_Sales"]) / historical_sales_period_weeks
    )
    weeks_cover = (
        float(product["Stock_Level"]) / baseline_weekly_sales
        if baseline_weekly_sales > 0
        else float("inf")
    )
    markdown_eligible = weeks_cover > horizon_weeks
    fixed_30 = _profit_curve(
        product,
        np.array([0.30]),
        horizon_weeks,
        unit_cost_rate,
        salvage_rate_of_cost,
        historical_sales_period_weeks,
    ).iloc[0]
    # Tolerance keeps least-squares rounding from flagging exactly covered stock.
    at_risk_unsold = bool(
        curve.loc[curve["markdown"].eq(0.0), "leftover_units"].iloc[0] > 1e-6
    )
    if not markdown_eligible:
        curve = curve.loc[curve["markdown"].eq(0.0)].reset_index(drop=True)
    best_index = int(curve["predicted_profit"].to_numpy().argmax())
    best = curve.iloc[best_index]
    if not markdown_eligible:
        action = ACTION_NO_MARKDOWN
    elif float(best["markdown"]) == 0.0:
        action = ACTION_HOLD
    elif horizon_weeks <= season_ending_weeks:
        action = ACTION_DEEPER
    else:
        action = ACTION_MARK_DOWN

    return {
        "SKU": product["Product_ID"],
        "Category": product["Category"],
        "Brand": product["Brand"],
        "original_price": float(product["Original_Price"]),
        "unit_cost": float(best["unit_cost"]),
        "salvage_value_per_unit": float(best["salvage_value_per_unit"]),
        "current_stock": float(product["Stock_Level"]),
        "horizon_weeks": horizon_weeks,
        "historical_sales_period_weeks": historical_sales_period_weeks,
        "weeks_cover": weeks_cover,
        "markdown_eligible_by_weeks_cover": markdown_eligible,
        "at_risk_unsold": at_risk_unsold,
        "demand_intercept_units_per_week": float(
            best["demand_intercept_units_per_week"]
        ),
        "demand_slope_units_per_week_per_markdown": float(
            best["demand_slope_units_per_week_per_markdown"]
        ),
        "demand_fit_r_squared": float(best["demand_fit_r_squared"]),
        "recommended_markdown": float(best["markdown"]),
        "recommended_price": float(best["price"]),
        "predicted_units_per_week": float(best["predicted_units_per_week"]),
        "uncapped_horizon_demand": float(best["uncapped_horizon_demand"]),
        "expected_sales": float(best["predicted_units_sold"]),
        "leftover_inventory": float(best["leftover_units"]),
        "sales_revenue": float(best["sales_revenue"]),
        "inventory_cost": float(best["inventory_cost"]),
        "salvage_proceeds": float(best["salvage_proceeds"]),
        "expected_margin": float(best["predicted_profit"]),
        "fixed_30_markdown": float(fixed_30["markdown"]),
        "fixed_30_expected_margin": float(fixed_30["predicted_profit"]),
        "profit_gain_vs_fixed_30": float(
            best["predicted_profit"] - fixed_30["predicted_profit"]
        ),
        "optimal_discount_reference": optimal_discount_reference,
        "recommended_minus_optimal_discount": (
            float(best["markdown"] - optimal_discount_reference)
            if optimal_discount_reference is not None
            else None
        ),
        "action": action,
        "recommendation_status": (
            "Scenario under assumptions - synthetic data; not validated"
        ),
        "candidate_markdowns_evaluated": int(len(markdown_grid)),
    }


def optimize_markdowns(
    frame: pd.DataFrame,
    horizon_weeks: int = HORIZON_WEEKS,
    unit_cost_rate: float = UNIT_COST_RATE,
    salvage_rate_of_cost: float = SALVAGE_RATE_OF_COST,
    max_markdown: float = MAX_MARKDOWN,
    markdown_step: float = MARKDOWN_STEP,
    historical_sales_period_weeks: int = 1,
    season_ending_weeks: int = SEASON_ENDING_WEEKS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return the maximum-profit markdown per product and deduplication counts."""
    if horizon_weeks <= 0:
        raise ValueError("horizon_weeks must be positive")
    if season_ending_weeks <= 0:
        raise ValueError("season_ending_weeks must be positive")
    if historical_sales_period_weeks <= 0:
        raise ValueError("historical_sales_period_weeks must be positive")
    if not 0 < unit_cost_rate < 1:
        raise ValueError("unit_cost_rate must be greater than 0 and less than 1")
    if not 0 <= salvage_rate_of_cost <= 1:
        raise ValueError("salvage_rate_of_cost must be between 0 and 1")
    if not 0 <= max_markdown < 1:
        raise ValueError("max_markdown must be at least 0 and less than 1")
    if markdown_step <= 0 or markdown_step > max_markdown and max_markdown > 0:
        raise ValueError("markdown_step must be positive and no greater than max_markdown")

    _validate_input(frame)
    source_rows = int(len(frame))
    analysis = frame.drop_duplicates().copy()
    duplicate_rows_removed = source_rows - len(analysis)
    if analysis["Product_ID"].duplicated().any():
        raise ValueError("Conflicting rows share a Product_ID after exact deduplication")
    has_optimal_discount = "Optimal Discount" in analysis.columns
    if has_optimal_discount:
        optimal_discount = pd.to_numeric(
            analysis["Optimal Discount"], errors="raise"
        )
        if (
            optimal_discount.isna().any()
            or not np.isfinite(optimal_discount.to_numpy(dtype=float)).all()
            or optimal_discount.lt(0).any()
            or optimal_discount.gt(1).any()
        ):
            raise ValueError("Optimal Discount comparison values must be in [0, 1]")
    else:
        optimal_discount = pd.Series(np.nan, index=analysis.index, dtype=float)
    analysis = analysis[INPUT_COLUMNS]

    candidate_count = int(np.floor(max_markdown / markdown_step + 1e-12))
    markdown_grid = np.arange(candidate_count + 1, dtype=float) * markdown_step
    if len(markdown_grid) == 0 or not np.isclose(markdown_grid[-1], max_markdown):
        markdown_grid = np.append(markdown_grid, max_markdown)

    recommendations = []
    for index, product in analysis.iterrows():
        comparison_value = (
            float(optimal_discount.loc[index]) if has_optimal_discount else None
        )
        recommendations.append(
            _simulate_product(
                product,
                markdown_grid,
                horizon_weeks,
                unit_cost_rate,
                salvage_rate_of_cost,
                optimal_discount_reference=comparison_value,
                historical_sales_period_weeks=historical_sales_period_weeks,
                season_ending_weeks=season_ending_weeks,
            )
        )
    results = pd.DataFrame(recommendations, columns=OUTPUT_COLUMNS)
    results = results.sort_values("SKU", kind="stable").reset_index(drop=True)
    audit: dict[str, object] = {
        "source_rows": source_rows,
        "exact_duplicate_rows_removed": int(duplicate_rows_removed),
        "analysis_products": int(len(analysis)),
        "horizon_weeks": horizon_weeks,
        "historical_sales_period_weeks": historical_sales_period_weeks,
        "unit_cost_rate_of_original_price": unit_cost_rate,
        "salvage_rate_of_unit_cost": salvage_rate_of_cost,
        "markdown_min": 0.0,
        "markdown_max": max_markdown,
        "markdown_step": markdown_step,
        "candidate_markdowns_per_product": int(len(markdown_grid)),
        "weeks_cover_threshold_rule": WEEKS_COVER_THRESHOLD_RULE,
        "season_ending_weeks": season_ending_weeks,
        "action_rules": ACTION_RULES,
        "action_counts": {
            str(action): int(count)
            for action, count in results["action"].value_counts().items()
        },
        "at_risk_unsold_count": int(results["at_risk_unsold"].sum()),
    }
    return results, audit


def markdown_economic_sensitivity(
    frame: pd.DataFrame,
    horizon_weeks: int = HORIZON_WEEKS,
    unit_cost_rates: tuple[float, ...] = SENSITIVITY_UNIT_COST_RATES,
    salvage_rates: tuple[float, ...] = SENSITIVITY_SALVAGE_RATES,
    historical_sales_period_weeks: int = 1,
) -> pd.DataFrame:
    """Count recommendation changes relative to the 60% cost / 20% salvage case."""
    if historical_sales_period_weeks <= 0:
        raise ValueError("historical_sales_period_weeks must be positive")
    _validate_input(frame)
    analysis = frame.drop_duplicates().copy()
    if analysis["Product_ID"].duplicated().any():
        raise ValueError("Conflicting rows share a Product_ID after exact deduplication")
    analysis = analysis[INPUT_COLUMNS]
    markdown_grid = np.arange(
        0.0, MAX_MARKDOWN + MARKDOWN_STEP / 2, MARKDOWN_STEP
    )
    default_rate = UNIT_COST_RATE
    default_salvage = SALVAGE_RATE_OF_COST
    fitted: list[
        tuple[int | str, pd.Series, float, float, float, np.ndarray, float, bool]
    ] = []
    for _, product in analysis.iterrows():
        intercept, slope, _ = _fit_demand_curve(
            product, historical_sales_period_weeks
        )
        demand = np.maximum(0.0, intercept + slope * markdown_grid) * horizon_weeks
        weekly_baseline = (
            float(product["Historical_Sales"]) / historical_sales_period_weeks
        )
        weeks_cover = (
            float(product["Stock_Level"]) / weekly_baseline
            if weekly_baseline > 0
            else float("inf")
        )
        eligible = weeks_cover > horizon_weeks
        fitted.append(
            (
                product["Product_ID"],
                product,
                intercept,
                slope,
                float(product["Original_Price"]),
                demand,
                weeks_cover,
                eligible,
            )
        )

    default_recommendations: dict[int | str, float] = {}
    for product_id, product, _, _, price, demand, _, eligible in fitted:
        stock = float(product["Stock_Level"])
        sold = np.minimum(demand, stock)
        leftovers = stock - sold
        unit_cost = price * default_rate
        salvage_value = unit_cost * default_salvage
        profit = (
            price * (1 - markdown_grid) * sold
            + salvage_value * leftovers
            - unit_cost * stock
        )
        default_recommendations[product_id] = (
            float(markdown_grid[int(profit.argmax())]) if eligible else 0.0
        )

    scenario_records: list[dict[str, object]] = []
    for cost_rate in unit_cost_rates:
        if not 0 < cost_rate < 1:
            raise ValueError("All unit_cost_rates must be between 0 and 1")
        for salvage_rate in salvage_rates:
            if not 0 <= salvage_rate <= 1:
                raise ValueError("All salvage_rates must be between 0 and 1")
            recommendations: list[float] = []
            for product_id, product, intercept, slope, price, demand, _, eligible in fitted:
                stock = float(product["Stock_Level"])
                sold = np.minimum(demand, stock)
                leftovers = stock - sold
                unit_cost = price * cost_rate
                salvage_value = unit_cost * salvage_rate
                profit = (
                    price * (1 - markdown_grid) * sold
                    + salvage_value * leftovers
                    - unit_cost * stock
                )
                markdown = (
                    float(markdown_grid[int(profit.argmax())]) if eligible else 0.0
                )
                recommendations.append(markdown)
            current = np.asarray(recommendations, dtype=float)
            baseline = np.asarray(
                [default_recommendations[product_id] for product_id, *_ in fitted],
                dtype=float,
            )
            changed = ~np.isclose(current, baseline, atol=1e-12, rtol=0)
            scenario_records.append(
                {
                    "scenario_label": "Scenario under assumptions - synthetic data",
                    "unit_cost_rate": cost_rate,
                    "salvage_rate_of_cost": salvage_rate,
                    "historical_sales_period_weeks": historical_sales_period_weeks,
                    "horizon_weeks": horizon_weeks,
                    "product_count": len(current),
                    "recommendation_changed_count_vs_60_20": int(changed.sum()),
                    "recommendation_changed_share_vs_60_20": (
                        float(changed.mean()) if len(changed) else 0.0
                    ),
                    "mean_recommended_markdown": (
                        float(current.mean()) if len(current) else None
                    ),
                    "median_recommended_markdown": (
                        float(np.median(current)) if len(current) else None
                    ),
                }
            )
    return pd.DataFrame(scenario_records)


def markdown_period_horizon_sensitivity(
    frame: pd.DataFrame,
    historical_sales_periods: tuple[int, ...] = PERIOD_HORIZON_VARIANTS,
    horizons: tuple[int, ...] = HORIZON_VARIANTS,
) -> pd.DataFrame:
    """Evaluate markdown picks across assumed historical-sales periods and horizons."""
    _validate_input(frame)
    analysis = frame.drop_duplicates().copy()
    if analysis["Product_ID"].duplicated().any():
        raise ValueError("Conflicting rows share a Product_ID after exact deduplication")
    analysis = analysis[INPUT_COLUMNS]
    grid = np.arange(0.0, MAX_MARKDOWN + MARKDOWN_STEP / 2, MARKDOWN_STEP)
    rows: list[dict[str, object]] = []
    for period_weeks in historical_sales_periods:
        if period_weeks <= 0:
            raise ValueError("Historical sales periods must be positive")
        products: list[tuple[pd.Series, float, float, float, float]] = []
        for _, product in analysis.iterrows():
            intercept, slope, r_squared = _fit_demand_curve(product, period_weeks)
            weekly_baseline = float(product["Historical_Sales"]) / period_weeks
            weeks_cover = (
                float(product["Stock_Level"]) / weekly_baseline
                if weekly_baseline > 0
                else float("inf")
            )
            products.append((product, intercept, slope, r_squared, weeks_cover))
        for horizon_weeks in horizons:
            if horizon_weeks <= 0:
                raise ValueError("Horizon weeks must be positive")
            picks: list[float] = []
            cover_values: list[float] = []
            for product, intercept, slope, _, weeks_cover in products:
                cover_values.append(weeks_cover)
                eligible = weeks_cover > horizon_weeks
                if not eligible:
                    picks.append(0.0)
                    continue
                weekly_demand = np.maximum(0.0, intercept + slope * grid)
                horizon_demand = weekly_demand * horizon_weeks
                stock = float(product["Stock_Level"])
                units_sold = np.minimum(horizon_demand, stock)
                leftover = stock - units_sold
                price = float(product["Original_Price"])
                unit_cost = price * UNIT_COST_RATE
                salvage = unit_cost * SALVAGE_RATE_OF_COST
                profit = (
                    price * (1 - grid) * units_sold
                    + salvage * leftover
                    - unit_cost * stock
                )
                picks.append(float(grid[int(profit.argmax())]))
            pick_values = np.asarray(picks)
            cover_array = np.asarray(cover_values)
            below_horizon = cover_array < horizon_weeks
            eligible_array = cover_array > horizon_weeks
            rows.append(
                {
                    "scenario_label": "Scenario under assumptions - synthetic data",
                    "historical_sales_period_weeks": period_weeks,
                    "horizon_weeks": horizon_weeks,
                    "product_count": len(analysis),
                    "weeks_cover_min": float(np.min(cover_array)),
                    "weeks_cover_p10": float(np.quantile(cover_array, 0.10)),
                    "weeks_cover_p25": float(np.quantile(cover_array, 0.25)),
                    "weeks_cover_median": float(np.quantile(cover_array, 0.50)),
                    "weeks_cover_p75": float(np.quantile(cover_array, 0.75)),
                    "weeks_cover_p90": float(np.quantile(cover_array, 0.90)),
                    "weeks_cover_max": float(np.max(cover_array)),
                    "stock_below_horizon_demand_count": int(below_horizon.sum()),
                    "stock_below_horizon_demand_share": float(below_horizon.mean()),
                    "markdown_eligible_by_cover_count": int(eligible_array.sum()),
                    "products_recommended_markdown_count": int(
                        (pick_values > 0).sum()
                    ),
                    "products_recommended_markdown_share": float(
                        (pick_values > 0).mean()
                    ),
                    "mean_recommended_markdown": float(pick_values.mean()),
                    "median_recommended_markdown": float(np.median(pick_values)),
                }
            )
    return pd.DataFrame(rows)


def _markdown_grid() -> np.ndarray:
    return np.arange(0.0, MAX_MARKDOWN + MARKDOWN_STEP / 2, MARKDOWN_STEP)


def _save_sku_profit_curves(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    evidence_dir: Path,
    horizon_weeks: int,
    historical_sales_period_weeks: int,
) -> pd.DataFrame:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    candidates = results.loc[
        ~np.isclose(results["recommended_markdown"], 0.30)
    ].sort_values(
        ["profit_gain_vs_fixed_30", "demand_slope_units_per_week_per_markdown", "SKU"],
        ascending=[False, False, True],
        kind="stable",
    ).head(5)
    selected = frame.drop_duplicates().set_index("Product_ID")
    grid = _markdown_grid()
    curve_frames = []
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    palette = ["#1769AA", "#D97732", "#7552A3", "#B84A55", "#6B7280"]
    for ax, (_, recommendation), color in zip(
        axes.flat,
        candidates.iterrows(),
        palette,
        strict=False,
    ):
        product = selected.loc[recommendation["SKU"]]
        curve = _profit_curve(
            product,
            grid,
            horizon_weeks,
            UNIT_COST_RATE,
            SALVAGE_RATE_OF_COST,
            historical_sales_period_weeks,
        )
        curve.insert(0, "SKU", recommendation["SKU"])
        curve.insert(
            1,
            "scenario_label",
            "Scenario under assumptions - synthetic data",
        )
        curve_frames.append(curve)
        ax.plot(
            curve["markdown"] * 100,
            curve["predicted_profit"],
            color=color,
            linewidth=2,
        )
        selected_markdown = recommendation["recommended_markdown"] * 100
        fixed_profit = float(recommendation["fixed_30_expected_margin"])
        ax.scatter(
            [selected_markdown],
            [recommendation["expected_margin"]],
            color=color,
            marker="o",
            label=f"Pick {selected_markdown:.0f}%",
            zorder=3,
        )
        ax.scatter(
            [30],
            [fixed_profit],
            color="#111827",
            marker="x",
            label="Fixed 30%",
            zorder=3,
        )
        optimal_reference = recommendation["optimal_discount_reference"]
        if pd.notna(optimal_reference):
            ax.axvline(
                float(optimal_reference) * 100,
                color="#8B5CF6",
                linestyle=":",
                linewidth=1.2,
                label="Optimal Discount reference",
            )
        ax.set_title(
            f"SKU {recommendation['SKU']} ({recommendation['Category']})"
        )
        ax.set_ylabel("Profit ($, scenario)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Markdown (%)")
    for ax in axes.flat[len(candidates) :]:
        ax.set_visible(False)
    figure.suptitle("Profit curves — scenario under assumptions (synthetic data)")
    figure.tight_layout()
    figure.savefig(evidence_dir / "sku_profit_curves.png", dpi=160)
    plt.close(figure)
    return (
        pd.concat(curve_frames, ignore_index=True)
        if curve_frames
        else pd.DataFrame()
    )


def _write_scenario_evidence(
    frame: pd.DataFrame,
    results: pd.DataFrame,
    sensitivity: pd.DataFrame,
    period_horizon: pd.DataFrame,
    evidence_dir: Path,
    horizon_weeks: int,
    historical_sales_period_weeks: int,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    curves = _save_sku_profit_curves(
        frame,
        results,
        evidence_dir,
        horizon_weeks,
        historical_sales_period_weeks,
    )
    curves.to_csv(evidence_dir / "sku_profit_curves.csv", index=False)
    sensitivity.to_csv(evidence_dir / "markdown_economic_sensitivity.csv", index=False)
    period_horizon.to_csv(
        evidence_dir / "markdown_period_horizon_sensitivity.csv", index=False
    )
    comparison_rows = results.loc[
        ~np.isclose(results["recommended_markdown"], 0.30)
    ]
    comparison_table = "\n".join(
        f"| {row.SKU} | {row.Category} | "
        f"{row.recommended_markdown:.0%} | {row.expected_margin:,.2f} | "
        f"{row.fixed_30_expected_margin:,.2f} | "
        f"{row.profit_gain_vs_fixed_30:,.2f} | "
        f"{row.optimal_discount_reference:.0%} |"
        for row in comparison_rows.head(5).itertuples(index=False)
    )
    sensitivity_table = "\n".join(
        f"| {row.unit_cost_rate:.0%} | {row.salvage_rate_of_cost:.0%} | "
        f"{row.recommendation_changed_count_vs_60_20:,} / {row.product_count:,} | "
        f"{row.recommendation_changed_share_vs_60_20:.1%} | "
        f"{row.mean_recommended_markdown:.1%} | "
        f"{row.median_recommended_markdown:.1%} |"
        for row in sensitivity.itertuples(index=False)
    )
    period_horizon_table = "\n".join(
        f"| {int(row.historical_sales_period_weeks)} | {int(row.horizon_weeks)} | "
        f"{row.stock_below_horizon_demand_count:,} "
        f"({row.stock_below_horizon_demand_share:.1%}) | "
        f"{row.markdown_eligible_by_cover_count:,} | "
        f"{row.products_recommended_markdown_count:,} "
        f"({row.products_recommended_markdown_share:.1%}) |"
        for row in period_horizon.itertuples(index=False)
    )
    period_one_horizon_four = period_horizon.loc[
        period_horizon["historical_sales_period_weeks"].eq(1)
        & period_horizon["horizon_weeks"].eq(4)
    ].iloc[0]
    weekly_cover_distribution = (
        f"| 1 week (weekly interpretation) | "
        f"{period_one_horizon_four.weeks_cover_min:.3f} | "
        f"{period_one_horizon_four.weeks_cover_p10:.3f} | "
        f"{period_one_horizon_four.weeks_cover_p25:.3f} | "
        f"{period_one_horizon_four.weeks_cover_median:.3f} | "
        f"{period_one_horizon_four.weeks_cover_p75:.3f} | "
        f"{period_one_horizon_four.weeks_cover_p90:.3f} | "
        f"{period_one_horizon_four.weeks_cover_max:.3f} |"
    )
    if len(comparison_rows):
        mean_profit_difference = float(
            comparison_rows["profit_gain_vs_fixed_30"].mean()
        )
        median_profit_difference = float(
            comparison_rows["profit_gain_vs_fixed_30"].median()
        )
        profit_comparison = (
            f"The recommendation differs from fixed 30% for {len(comparison_rows):,} "
            f"of {len(results):,} products. Among only those differing products, "
            f"mean per-product profit difference (pick minus fixed 30%) is "
            f"${mean_profit_difference:,.2f}; median is ${median_profit_difference:,.2f}. "
            "No portfolio-wide profit gap is reported."
        )
    else:
        profit_comparison = (
            f"The recommendation differs from fixed 30% for 0 of {len(results):,} "
            "products, so there are no differing cases for profit comparison."
        )
    optimal_reference_count = int(results["optimal_discount_reference"].notna().sum())
    optimal_mae = (
        float(results["recommended_minus_optimal_discount"].abs().mean())
        if optimal_reference_count
        else float("nan")
    )
    optimal_summary = (
        f"The mean absolute markdown difference from the supplied `Optimal Discount` "
        f"comparison column is {optimal_mae:.1%} across {optimal_reference_count:,} "
        "products. `Optimal Discount` is not used to fit demand or choose the "
        "simulated markdown."
        if optimal_reference_count
        else "No `Optimal Discount` comparison column was present in the input."
    )
    report = f"""# Module 4 evidence — scenario under assumptions

> **Scenario under assumptions — synthetic data.** D2 identifies itself as
> synthetic. The profits and markdowns below are scenario outputs, not real
> retail outcomes, causal effects, or validated recommendations.

## Method and assumptions

The linear weekly demand curve for each product uses the four matched
`(Markdown_k, Sales_After_Mk)` pairs plus `(0, Historical_Sales)`. For each
markdown from 0% to 50% in 10-point increments, it projects demand over
{horizon_weeks} weeks, caps units sold at stock, charges cost on all opening
stock, and credits salvage on leftover units. The historical-sales period is
assumed to be {historical_sales_period_weeks} week(s); default unit cost is 60%
of original price and salvage is 20% of unit cost.

**Markdown eligibility rule:** recommend a markdown only when stock cover,
`Stock_Level / (Historical_Sales / assumed_period_weeks)`, strictly exceeds
the simulated horizon. Otherwise hold at 0%, since baseline demand covers
opening stock within the horizon.

`Optimal Discount` appears only as a comparison column/reference marker. It
does not enter the demand fit, feature set, or profit maximization.

## Pick versus fixed 30% — first five differing SKUs

| Product ID | Category | Pick | Pick profit ($) | Fixed 30% profit ($) | Pick gain ($) | `Optimal Discount` reference only |
|---:|---|---:|---:|---:|---:|---:|
{comparison_table if len(comparison_rows) else "| No differing products | — | — | — | — | — | — |"}

{profit_comparison} {optimal_summary}

## Five SKU profit curves

The plotted five are the products with the largest per-product profit
improvement over fixed 30% among differing cases, under 60% cost / 20%
salvage. Markers show the selected markdown and fixed 30% comparison. The
purple vertical reference is the supplied
`Optimal Discount` value and is comparison-only.

![Five SKU profit curves — scenario under assumptions](sku_profit_curves.png)

The full plotted curves are saved to
`outputs/module4/evidence/sku_profit_curves.csv`.

## Economic sensitivity

For each cost/salvage combination in the 13-week historical-sales / 8-week
horizon scenario, “changed” counts products whose best grid markdown differs
from the default 60% cost / 20% salvage recommendation.

| Unit cost (% of original price) | Salvage (% of unit cost) | Changed recommendations vs 60/20 | Share changed | Mean picked markdown | Median picked markdown |
|---:|---:|---:|---:|---:|---:|
{sensitivity_table}

The complete 3×3 table is also in
`outputs/module4/evidence/markdown_economic_sensitivity.csv`.

## Historical-sales period and horizon sensitivity

`weeks_cover` is stock divided by assumed weekly baseline demand. For a
13- or 26-week historical period, weekly demand is `Historical_Sales` divided
by that period. “Stock below horizon demand” compares opening stock to
projected baseline units for the horizon.

| Assumed historical period (weeks) | Horizon (weeks) | Stock below horizon demand | Cover-eligible products | Products with markdown |
|---:|---:|---:|---:|---:|
{period_horizon_table}

| Sales-period assumption | Min | P10 | P25 | Median | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
{weekly_cover_distribution}

Under the weekly interpretation, stock is below baseline demand over four
weeks for {period_one_horizon_four.stock_below_horizon_demand_count:,} products
({period_one_horizon_four.stock_below_horizon_demand_share:.1%}).
These results are conditional on the period assumption; the source does not
define the sales measurement periods.
"""
    (evidence_dir / "markdown_scenario_evidence.md").write_text(
        report,
        encoding="utf-8",
    )


def build_markdown_recommendations(
    input_path: Path,
    output_dir: Path,
    horizon_weeks: int = HORIZON_WEEKS,
    evidence_dir: Path = Path("outputs/module4/evidence"),
    historical_sales_period_weeks: int = 1,
    season_ending_weeks: int = SEASON_ENDING_WEEKS,
) -> pd.DataFrame:
    """Read the D2 CSV, optimize all products, and write results plus audit."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Markdown input file not found: {input_path}")
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    frame = pd.read_csv(input_path)
    results, audit = optimize_markdowns(
        frame,
        horizon_weeks=horizon_weeks,
        historical_sales_period_weeks=historical_sales_period_weeks,
        season_ending_weeks=season_ending_weeks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity = markdown_economic_sensitivity(
        frame,
        horizon_weeks=8,
        historical_sales_period_weeks=13,
    )
    period_horizon = markdown_period_horizon_sensitivity(frame)
    evidence_results, _ = optimize_markdowns(
        frame,
        horizon_weeks=8,
        historical_sales_period_weeks=13,
    )
    _write_scenario_evidence(
        frame,
        evidence_results,
        sensitivity,
        period_horizon,
        evidence_dir,
        8,
        13,
    )
    result_path = output_dir / "markdown_recommendations.csv"
    audit_path = output_dir / "markdown_recommendations_audit.json"
    results.to_csv(result_path, index=False)
    markdown_summary = (
        {
            "mean": float(results["recommended_markdown"].mean()),
            "median": float(results["recommended_markdown"].median()),
            "min": float(results["recommended_markdown"].min()),
            "max": float(results["recommended_markdown"].max()),
        }
        if not results.empty
        else {"mean": None, "median": None, "min": None, "max": None}
    )
    audit_data: dict[str, object] = {
        "source_file": str(input_path),
        "source_sha256": source_hash,
        "assumptions": {
            "historical_sales": (
                "The default CSV uses Historical_Sales as units/week; sensitivity "
                "scenarios also use assumed 13- and 26-week measurement periods."
            ),
            "matched_sales": (
                "Each Sales_After_Mk is treated as units from the same assumed "
                "historical-sales period at Markdown_k."
            ),
            "unit_cost": "Scenario under assumptions: unit cost is 60% of Original_Price.",
            "salvage": "Scenario under assumptions: salvage value per leftover unit is 20% of unit cost.",
            "horizon": f"Demand is projected over {horizon_weeks} weeks.",
            "demand_model": (
                "Per-product ordinary least-squares linear fit to the four "
                "(Markdown_k, Sales_After_Mk) pairs plus (0, Historical_Sales); "
                "negative predicted demand is floored at zero."
            ),
            "stock_cap": "Horizon units sold are capped at Stock_Level.",
            "profit": (
                "Sales revenue + salvage proceeds on leftover opening stock "
                "- unit cost times all opening stock."
            ),
            "optimization_grid": (
                f"Enumerate markdowns from 0% to {MAX_MARKDOWN:.0%} inclusive "
                f"in {MARKDOWN_STEP:.0%} increments; choose maximum predicted profit. "
                "Ties select the lower markdown."
            ),
            "weeks_cover_markdown_rule": WEEKS_COVER_THRESHOLD_RULE,
            "duplicate_handling": "Scenario under assumptions: remove exact duplicate source rows; reject conflicting Product_ID rows.",
            "optimal_discount_comparison": (
                "The source Optimal Discount column is output only for comparison; "
                "it is excluded from demand fitting and optimization."
            ),
        },
        **audit,
        "scenario_label": "Scenario under assumptions - synthetic data",
        "period_horizon_sensitivity_file": (
            "markdown_period_horizon_sensitivity.csv"
        ),
        "recommended_markdown_summary": markdown_summary,
        "recommendations_file": result_path.name,
        "evidence_directory": str(evidence_dir),
    }
    audit_path.write_text(json.dumps(audit_data, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/markdown/SYNTHETIC Markdown Dataset.csv"),
        help="D2 synthetic markdown CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/module4"),
        help="Directory for recommendations and audit JSON",
    )
    parser.add_argument(
        "--horizon-weeks",
        type=int,
        default=HORIZON_WEEKS,
        help="Assumed demand horizon in weeks",
    )
    parser.add_argument(
        "--historical-sales-period-weeks",
        type=int,
        default=1,
        help="Assumed period length for Historical_Sales and Sales_After_Mk",
    )
    parser.add_argument(
        "--season-ending-weeks",
        type=int,
        default=SEASON_ENDING_WEEKS,
        help="Horizons at or below this many weeks count as season ending soon",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("outputs/module4/evidence"),
        help="Directory for profit curves, sensitivity results, and evidence report",
    )
    args = parser.parse_args()
    results = build_markdown_recommendations(
        args.input,
        args.output_dir,
        horizon_weeks=args.horizon_weeks,
        evidence_dir=args.evidence_dir,
        historical_sales_period_weeks=args.historical_sales_period_weeks,
        season_ending_weeks=args.season_ending_weeks,
    )
    print(f"Products optimized: {len(results):,}")
    print(
        "Recommended markdown distribution:\n"
        + results["recommended_markdown"].value_counts().sort_index().to_string()
    )
    print("Actions:\n" + results["action"].value_counts().to_string())
    print(f"Recommendations: {args.output_dir / 'markdown_recommendations.csv'}")
    print(f"Audit: {args.output_dir / 'markdown_recommendations_audit.json'}")
    print(f"Evidence: {args.evidence_dir / 'markdown_scenario_evidence.md'}")


if __name__ == "__main__":
    main()
