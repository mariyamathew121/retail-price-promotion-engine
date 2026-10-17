"""Estimate department-level pooled price elasticities with high-dimensional FE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf

from retail_engine.module2 import WEEK_START_SQL

MIN_WEEKS = 104
MIN_POSITIVE_WEEKS = 52
MIN_PRICE_LEVELS = 3
MIN_PRICE_RANGE = 0.05

ELIGIBLE_SERIES_SQL = f"""
WITH weekly AS (
    SELECT d.item_id, d.store_id, d.dept_id,
           {WEEK_START_SQL} AS week_start,
           SUM(d.units_sold)::DOUBLE AS units_sold,
           AVG(d.price)::DOUBLE AS price
    FROM read_parquet(?) AS d
    GROUP BY d.item_id, d.store_id, d.dept_id, week_start
)
SELECT item_id, store_id, dept_id,
       COUNT(*)::INTEGER AS weeks_observed,
       COUNT(*) FILTER (WHERE units_sold > 0)::INTEGER AS positive_weeks,
       COUNT(DISTINCT price)::INTEGER AS price_levels,
       SUM(units_sold)::DOUBLE AS total_units,
       (MAX(price) / MIN(price) - 1)::DOUBLE AS price_range_pct
FROM weekly
GROUP BY item_id, store_id, dept_id
HAVING COUNT(*) >= ?
   AND COUNT(*) FILTER (WHERE units_sold > 0) >= ?
   AND COUNT(DISTINCT price) >= ?
   AND MAX(price) / MIN(price) - 1 >= ?
ORDER BY dept_id, total_units DESC, item_id, store_id
"""

WEEKLY_SAMPLE_SQL = f"""
SELECT d.item_id, d.store_id, d.dept_id,
       {WEEK_START_SQL} AS week_start,
       SUM(d.units_sold)::DOUBLE AS units_sold,
       AVG(d.price)::DOUBLE AS price
FROM read_parquet(?) AS d
INNER JOIN shortlisted AS s
    ON d.item_id = s.item_id AND d.store_id = s.store_id
GROUP BY d.item_id, d.store_id, d.dept_id, week_start
ORDER BY d.dept_id, d.item_id, d.store_id, week_start
"""


def _select_stratified_series(
    eligible: pd.DataFrame, max_series: int
) -> pd.DataFrame:
    if max_series < 1:
        raise ValueError("max_series must be at least 1")
    if eligible.empty:
        raise ValueError("No item-store series meet the eligibility thresholds")

    departments = sorted(eligible["dept_id"].unique())
    base, remainder = divmod(max_series, len(departments))
    selected_groups: list[pd.DataFrame] = []
    selected_keys: set[tuple[str, str]] = set()

    for index, department in enumerate(departments):
        count = base + (index < remainder)
        group = eligible[eligible["dept_id"] == department].head(count)
        selected_groups.append(group)
        selected_keys.update(zip(group["item_id"], group["store_id"]))

    selected_count = sum(len(group) for group in selected_groups)
    if selected_count < max_series:
        remaining = eligible[
            ~eligible.apply(
                lambda row: (row["item_id"], row["store_id"]) in selected_keys,
                axis=1,
            )
        ].head(max_series - selected_count)
        selected_groups.append(remaining)

    selected = pd.concat(selected_groups, ignore_index=True)
    return selected.sort_values(
        ["dept_id", "total_units", "item_id", "store_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _fit_department(group: pd.DataFrame) -> dict[str, object]:
    model_data = group.loc[
        (group["units_sold"] > 0) & (group["price"] > 0)
    ].copy()
    if model_data.empty:
        raise ValueError(f"No positive-sales observations in {group['dept_id'].iloc[0]}")

    model_data["log_units"] = np.log(model_data["units_sold"])
    model_data["log_price"] = np.log(model_data["price"])
    model_data["item_store"] = (
        model_data["item_id"].astype(str) + "|" + model_data["store_id"].astype(str)
    )
    model_data["store_week"] = (
        model_data["store_id"].astype(str)
        + "|"
        + pd.to_datetime(model_data["week_start"]).dt.strftime("%Y-%m-%d")
    )

    model = pf.feols(
        "log_units ~ log_price | item_store + store_week",
        data=model_data,
        vcov={"CRV1": "item_store"},
    )
    term = model.tidy().loc["log_price"]
    estimate = float(term["Estimate"])
    standard_error = float(term["Std. Error"])
    return {
        "dept_id": str(group["dept_id"].iloc[0]),
        "elasticity": estimate,
        "standard_error": standard_error,
        "ci_lower": float(term["2.5%"]),
        "ci_upper": float(term["97.5%"]),
        "p_value": float(term["Pr(>|t|)"]),
        "positive_sales_weekly_rows": int(len(model_data)),
        "candidate_item_store_clusters": int(model_data["item_store"].nunique()),
        "price_change_weeks": int(
            (
                group.sort_values(["item_id", "store_id", "week_start"])
                .groupby(["item_id", "store_id"], sort=False)["price"]
                .diff()
                .abs()
                .gt(1e-9)
            ).sum()
        ),
        "zero_sales_weeks_excluded": int((group["units_sold"] == 0).sum()),
        "zero_sales_share": float((group["units_sold"] == 0).mean()),
    }


def run_pooled_elasticity(
    input_path: Path, output_dir: Path, max_series: int = 2_000
) -> pd.DataFrame:
    con = duckdb.connect()
    eligible = con.execute(
        ELIGIBLE_SERIES_SQL,
        [
            str(input_path),
            MIN_WEEKS,
            MIN_POSITIVE_WEEKS,
            MIN_PRICE_LEVELS,
            MIN_PRICE_RANGE,
        ],
    ).df()
    selected = _select_stratified_series(eligible, max_series)
    con.register("shortlisted", selected[["item_id", "store_id"]])
    weekly = con.execute(WEEKLY_SAMPLE_SQL, [str(input_path)]).df()
    con.close()

    results = pd.DataFrame(
        [
            _fit_department(group)
            for _, group in weekly.groupby("dept_id", sort=True)
        ]
    )
    results.insert(1, "selected_series", results["dept_id"].map(
        selected.groupby("dept_id").size()
    ).astype(int))
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "pooled_price_elasticities.csv", index=False)

    audit = {
        "input": str(input_path),
        "model": "log(units_sold) ~ log(price) | item-store FE + store-week FE",
        "standard_errors": "CRV1 clustered by item-store",
        "singleton_fixed_effects": (
            "PyFixest drops singleton fixed-effect observations during estimation."
        ),
        "zero_sales": "Zero-sales weeks excluded because log(units_sold) is undefined.",
        "sample_strategy": (
            "Top eligible series by total units within each department, allocated "
            "as evenly as possible across departments."
        ),
        "eligibility": {
            "minimum_weeks": MIN_WEEKS,
            "minimum_positive_weeks": MIN_POSITIVE_WEEKS,
            "minimum_price_levels": MIN_PRICE_LEVELS,
            "minimum_price_range": MIN_PRICE_RANGE,
        },
        "requested_max_series": max_series,
        "eligible_series": int(len(eligible)),
        "selected_series": int(len(selected)),
        "selected_series_by_department": {
            str(key): int(value)
            for key, value in selected.groupby("dept_id").size().items()
        },
        "results": results.to_dict(orient="records"),
    }
    (output_dir / "pooled_price_elasticities_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate pooled department elasticities with item-store and store-week fixed effects."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/module2_pooled"),
    )
    parser.add_argument("--max-series", type=int, default=2_000)
    args = parser.parse_args()

    results = run_pooled_elasticity(args.input, args.output_dir, args.max_series)
    print(results.to_string(index=False))
    print(f"\nSaved results and audit to {args.output_dir}")


if __name__ == "__main__":
    main()
