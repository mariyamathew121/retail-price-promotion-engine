"""Daily-grain elasticity check using the brief's controls (weekday, month, SNAP, event, promo).

The primary Module 2 models run at weekly grain because M5 prices change weekly, so
weekday cannot enter them and month is only approximated by seasonality curves. This
module re-estimates the same item-store series on daily rows, adding weekday and month
indicators, and compares the result with the weekly estimates. Store and category are
constant within one item-store series, so they cannot act as controls here; they are
absorbed by the series itself (and by fixed effects in the pooled models).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retail_engine.module2 import (
    _crossfit_dml,
    _interval,
    _ols_log_log,
    classify_elasticity,
)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = list(range(1, 13))
# Daily residuals within a price week are strongly dependent, so the Newey-West lag is
# four weeks of days (28), matching the weekly models' four-week lag.
HAC_LAG_DAYS = 28
DAILY_SQL = """
SELECT d.item_id, d.store_id, CAST(d.date AS DATE) AS date,
       d.units_sold::DOUBLE AS units_sold, d.price::DOUBLE AS price,
       d.snap_flag::DOUBLE AS snap, d.promo_flag::DOUBLE AS promo,
       (CASE WHEN d.event_name IS NOT NULL AND d.event_name <> ''
             THEN 1 ELSE 0 END)::DOUBLE AS event,
       d.weekday AS weekday, d.month::INTEGER AS month
FROM read_parquet(?) AS d
INNER JOIN shortlisted AS s
    ON d.item_id = s.item_id AND d.store_id = s.store_id
ORDER BY d.item_id, d.store_id, d.date
"""
RESULT_COLUMNS = [
    "item_id",
    "store_id",
    "daily_elasticity",
    "daily_conf_interval",
    "daily_ci_lower",
    "daily_ci_upper",
    "daily_naive_elasticity",
    "daily_ols_elasticity",
    "daily_demand_type",
    "days_used",
    "fit_status",
]


def daily_control_matrix(daily: pd.DataFrame) -> np.ndarray:
    """SNAP, event and inferred-promo flags plus weekday and month indicators."""
    weekday = pd.Categorical(daily["weekday"], categories=WEEKDAYS)
    month = pd.Categorical(daily["month"], categories=MONTHS)
    weekday_dummies = pd.get_dummies(weekday, dtype=np.float64).iloc[:, 1:]
    month_dummies = pd.get_dummies(month, dtype=np.float64).iloc[:, 1:]
    flags = daily[["snap", "event", "promo"]].to_numpy(dtype=np.float64)
    return np.column_stack([flags, weekday_dummies.to_numpy(), month_dummies.to_numpy()])


def _estimate_series(
    daily: pd.DataFrame, *, folds: int, trees: int, seed: int
) -> dict[str, object]:
    daily = daily.sort_values("date", kind="stable").reset_index(drop=True)
    row: dict[str, object] = {
        "item_id": daily["item_id"].iloc[0],
        "store_id": daily["store_id"].iloc[0],
        "daily_elasticity": np.nan,
        "daily_conf_interval": "",
        "daily_ci_lower": np.nan,
        "daily_ci_upper": np.nan,
        "daily_naive_elasticity": np.nan,
        "daily_ols_elasticity": np.nan,
        "daily_demand_type": "Unclear",
        "days_used": len(daily),
        "fit_status": "not_estimable",
    }
    if len(daily) < folds * 5:
        row["fit_status"] = "not_estimable: too few observed days"
        return row
    dates = pd.to_datetime(daily["date"])
    day_index = (dates - dates.min()).dt.days.to_numpy(dtype=np.int64)
    controls = daily_control_matrix(daily)
    log_price = np.log(daily["price"].to_numpy(dtype=np.float64))
    log_units = np.log1p(daily["units_sold"].to_numpy(dtype=np.float64))
    try:
        estimate, se = _crossfit_dml(
            log_price,
            log_units,
            controls,
            day_index,
            folds=folds,
            trees=trees,
            seed=seed,
            max_lag=HAC_LAG_DAYS,
        )
        ols, _ = _ols_log_log(log_price, log_units, controls, day_index, HAC_LAG_DAYS)
        naive, _ = _ols_log_log(
            log_price, log_units, np.zeros((len(daily), 0)), day_index, HAC_LAG_DAYS
        )
    except ValueError as error:
        row["fit_status"] = f"not_estimable: {error}"
        return row
    lower, upper = _interval(estimate, se)
    demand_type, _ = classify_elasticity(estimate, lower, upper)
    row.update(
        {
            "daily_elasticity": estimate,
            "daily_conf_interval": f"[{lower:.3f}, {upper:.3f}]",
            "daily_ci_lower": lower,
            "daily_ci_upper": upper,
            "daily_naive_elasticity": naive,
            "daily_ols_elasticity": ols,
            "daily_demand_type": demand_type,
            "fit_status": "estimated",
        }
    )
    return row


def estimate_daily_elasticities(
    daily: pd.DataFrame, *, folds: int = 5, trees: int = 50, seed: int = 2026
) -> pd.DataFrame:
    """Estimate each item-store series in `daily` (columns as in DAILY_SQL)."""
    rows = [
        _estimate_series(group, folds=folds, trees=trees, seed=seed)
        for _, group in daily.groupby(["item_id", "store_id"], sort=True)
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def compare_with_weekly(daily_results: pd.DataFrame, weekly_results: pd.DataFrame) -> pd.DataFrame:
    """Join daily estimates to the weekly per-SKU results for the same series."""
    weekly = weekly_results[
        ["item_id", "store_id", "elasticity", "ci_lower", "ci_upper", "demand_type"]
    ].rename(
        columns={
            "elasticity": "weekly_elasticity",
            "ci_lower": "weekly_ci_lower",
            "ci_upper": "weekly_ci_upper",
            "demand_type": "weekly_demand_type",
        }
    )
    return daily_results.merge(weekly, on=["item_id", "store_id"], how="left")


def comparison_markdown(compared: pd.DataFrame) -> str:
    estimated = compared.loc[compared["fit_status"].eq("estimated")]
    if estimated.empty:
        return "# Module 2 daily-grain check\n\nNo series produced a daily estimate.\n"
    difference = estimated["daily_elasticity"] - estimated["weekly_elasticity"]
    sign_agree = (np.sign(estimated["daily_elasticity"]) == np.sign(estimated["weekly_elasticity"])).mean()
    type_agree = (estimated["daily_demand_type"] == estimated["weekly_demand_type"]).mean()
    both_cis_overlap = (
        (estimated["daily_ci_lower"] <= estimated["weekly_ci_upper"])
        & (estimated["weekly_ci_lower"] <= estimated["daily_ci_upper"])
    ).mean()
    width = (estimated["daily_ci_upper"] - estimated["daily_ci_lower"]).median()
    weekly_width = (estimated["weekly_ci_upper"] - estimated["weekly_ci_lower"]).median()
    counts = estimated["daily_demand_type"].value_counts()
    lines = [
        "# Module 2 daily-grain check (brief's control list)",
        "",
        "The primary models are weekly because M5 prices change weekly. This check "
        "re-estimates the same series on daily rows with weekday and month indicators, "
        "SNAP, event and the inferred promotion flag as controls (outcome `log1p(units)`, "
        "treatment `log(price)`, cross-fitted Extra Trees DML, Newey-West lag 28 days). "
        "Store and category are constant within an item-store series, so they cannot be "
        "controls here.",
        "",
        f"- Series estimated: {len(estimated)} of {len(compared)}",
        f"- Median daily DML elasticity: {estimated['daily_elasticity'].median():.3f} "
        f"(weekly: {estimated['weekly_elasticity'].median():.3f})",
        f"- Median absolute difference from the weekly estimate: {difference.abs().median():.3f}",
        f"- Correlation with the weekly estimate: "
        f"{estimated['daily_elasticity'].corr(estimated['weekly_elasticity']):.2f}",
        f"- Same sign as the weekly estimate: {sign_agree:.0%}",
        f"- Same demand type as the weekly estimate: {type_agree:.0%}",
        f"- Weekly and daily 95% intervals overlap: {both_cis_overlap:.0%}",
        f"- Median interval width: daily {width:.3f}, weekly {weekly_width:.3f}",
        "- Daily demand types: "
        + ", ".join(f"{count} {name}" for name, count in counts.items()),
        "",
        "Daily rows repeat each week's price seven times, so the extra rows add weekday "
        "and month detail to the outcome but no new price variation; the intervals use "
        "a 28-day Newey-West lag to account for that, and should not be read as "
        "more precise than the weekly ones. Observational estimates; see assumptions.md.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clean_retail_data.parquet")
    )
    parser.add_argument(
        "--weekly-results",
        type=Path,
        default=Path("data/processed/module2/price_elasticities.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/module2/daily_grain_check.csv")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/module2/daily_grain_check.md")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=50)
    args = parser.parse_args()

    weekly = pd.read_csv(args.weekly_results)
    shortlisted = weekly[["item_id", "store_id"]]
    connection = duckdb.connect()
    connection.register("shortlisted", shortlisted)
    daily = connection.execute(DAILY_SQL, [str(args.input)]).df()
    results = compare_with_weekly(
        estimate_daily_elasticities(daily, folds=args.folds, trees=args.trees), weekly
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(comparison_markdown(results), encoding="utf-8")
    print(comparison_markdown(results))


if __name__ == "__main__":
    main()
