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
    capacities = {
        department: int((eligible["dept_id"] == department).sum())
        for department in departments
    }
    allocations = {
        department: min(base + (index < remainder), capacities[department])
        for index, department in enumerate(departments)
    }
    deficit = max_series - sum(allocations.values())
    while deficit:
        available_departments = [
            department
            for department in departments
            if allocations[department] < capacities[department]
        ]
        if not available_departments:
            break
        lowest_count = min(allocations[department] for department in available_departments)
        for department in available_departments:
            if allocations[department] == lowest_count:
                allocations[department] += 1
                deficit -= 1
                if deficit == 0:
                    break

    selected_groups = [
        eligible[eligible["dept_id"] == department].head(allocations[department])
        for department in departments
    ]
    selected = pd.concat(selected_groups, ignore_index=True)
    return selected.sort_values(
        ["dept_id", "total_units", "item_id", "store_id"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _fit_department(group: pd.DataFrame) -> dict[str, object]:
    all_week_data = group.loc[
        (group["units_sold"] >= 0) & (group["price"] > 0)
    ].copy()
    model_data = all_week_data.loc[
        (group["units_sold"] > 0) & (group["price"] > 0)
    ].copy()
    if model_data.empty:
        raise ValueError(f"No positive-sales observations in {group['dept_id'].iloc[0]}")

    model_data["log_units"] = np.log(model_data["units_sold"])
    model_data["log_price"] = np.log(model_data["price"])
    all_week_data["log_price"] = np.log(all_week_data["price"])
    for frame in (model_data, all_week_data):
        frame["item_store"] = (
            frame["item_id"].astype(str) + "|" + frame["store_id"].astype(str)
        )
        frame["store_week"] = (
            frame["store_id"].astype(str)
            + "|"
            + pd.to_datetime(frame["week_start"]).dt.strftime("%Y-%m-%d")
        )

    feols_model = pf.feols(
        "log_units ~ log_price | item_store + store_week",
        data=model_data,
        vcov={"CRV1": "item_store"},
    )
    feols_term = feols_model.tidy().loc["log_price"]
    ppml_model = pf.fepois(
        "units_sold ~ log_price | item_store + store_week",
        data=all_week_data,
        vcov={"CRV1": "item_store"},
    )
    ppml_term = ppml_model.tidy().loc["log_price"]
    feols_estimate = float(feols_term["Estimate"])
    ppml_estimate = float(ppml_term["Estimate"])
    return {
        "store_id": (
            str(group["store_id"].iloc[0])
            if group["store_id"].nunique() == 1
            else "ALL"
        ),
        "dept_id": str(group["dept_id"].iloc[0]),
        "feols_elasticity": feols_estimate,
        "feols_standard_error": float(feols_term["Std. Error"]),
        "feols_ci_lower": float(feols_term["2.5%"]),
        "feols_ci_upper": float(feols_term["97.5%"]),
        "feols_p_value": float(feols_term["Pr(>|t|)"]),
        "ppml_elasticity": ppml_estimate,
        "ppml_standard_error": float(ppml_term["Std. Error"]),
        "ppml_ci_lower": float(ppml_term["2.5%"]),
        "ppml_ci_upper": float(ppml_term["97.5%"]),
        "ppml_p_value": float(ppml_term["Pr(>|t|)"]),
        "feols_positive_sales_weekly_rows": int(len(model_data)),
        "ppml_all_weekly_rows": int(len(all_week_data)),
        "candidate_item_store_clusters": int(model_data["item_store"].nunique()),
        "ppml_candidate_item_store_clusters": int(all_week_data["item_store"].nunique()),
        "price_change_weeks": int(
            (
                all_week_data.sort_values(["item_id", "store_id", "week_start"])
                .groupby(["item_id", "store_id"], sort=False)["price"]
                .diff()
                .abs()
                .gt(1e-9)
            ).sum()
        ),
        "zero_sales_weeks_excluded_from_feols": int(
            (all_week_data["units_sold"] == 0).sum()
        ),
        "zero_sales_share": float((all_week_data["units_sold"] == 0).mean()),
    }


def _pooled_action(
    elasticity: float, lower: float, upper: float
) -> tuple[str, str]:
    if lower <= 0 <= upper or elasticity >= 0:
        return "Unclear", "Flag for review; take no pricing action"
    if elasticity < -1:
        return "Price-sensitive", "Test a controlled discount or promotion"
    return "Less sensitive", "Protect price or test a small increase"


def _build_item_store_recommendations(
    selected: pd.DataFrame,
    department_results: pd.DataFrame,
    dml_results: pd.DataFrame,
) -> pd.DataFrame:
    recommendations = selected.merge(
        department_results[
            [
                "dept_id",
                "ppml_elasticity",
                "ppml_ci_lower",
                "ppml_ci_upper",
            ]
        ],
        on="dept_id",
        how="left",
        validate="many_to_one",
    )
    if dml_results.empty:
        recommendations["dml_elasticity"] = np.nan
        recommendations["dml_demand_type"] = pd.NA
    else:
        dml_columns = dml_results[
            ["item_id", "store_id", "elasticity", "demand_type"]
        ].rename(
            columns={
                "elasticity": "dml_elasticity",
                "demand_type": "dml_demand_type",
            }
        )
        recommendations = recommendations.merge(
            dml_columns,
            on=["item_id", "store_id"],
            how="left",
            validate="one_to_one",
        )

    actions = [
        _pooled_action(estimate, lower, upper)
        for estimate, lower, upper in zip(
            recommendations["ppml_elasticity"],
            recommendations["ppml_ci_lower"],
            recommendations["ppml_ci_upper"],
        )
    ]
    recommendations["pooled_demand_type"] = [action[0] for action in actions]
    recommendations["pricing_recommendation"] = [action[1] for action in actions]
    has_dml = recommendations["dml_demand_type"].notna()
    recommendations["dml_review_flag"] = np.select(
        [
            ~has_dml,
            recommendations["dml_demand_type"]
            != recommendations["pooled_demand_type"],
        ],
        ["No individual DML estimate", "DML class differs from pooled default"],
        default="DML class agrees with pooled default",
    )
    return recommendations[
        [
            "item_id",
            "store_id",
            "dept_id",
            "ppml_elasticity",
            "ppml_ci_lower",
            "ppml_ci_upper",
            "pooled_demand_type",
            "pricing_recommendation",
            "dml_elasticity",
            "dml_demand_type",
            "dml_review_flag",
        ]
    ]


def run_pooled_elasticity(
    input_path: Path,
    output_dir: Path,
    max_series: int = 2_000,
    by_store: bool = False,
    dml_results_path: Path | None = None,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_dir / "selected_series.csv", index=False)

    results = pd.DataFrame(
        [
            _fit_department(group)
            for _, group in weekly.groupby("dept_id", sort=True)
        ]
    )
    results.insert(1, "selected_series", results["dept_id"].map(
        selected.groupby("dept_id").size()
    ).astype(int))
    results.to_csv(output_dir / "pooled_price_elasticities.csv", index=False)

    dml_results = pd.DataFrame()
    dml_source = dml_results_path or input_path.parent / "module2" / "price_elasticities.csv"
    if dml_source.is_file():
        dml_results = pd.read_csv(dml_source)
        required_dml_columns = {"item_id", "store_id", "elasticity", "demand_type"}
        if not required_dml_columns.issubset(dml_results.columns):
            raise ValueError(
                f"DML results must contain columns: {sorted(required_dml_columns)}"
            )
    recommendations = _build_item_store_recommendations(
        selected, results, dml_results
    )
    recommendations.to_csv(
        output_dir / "item_store_pricing_defaults.csv",
        index=False,
    )

    store_results = None
    if by_store:
        store_results = pd.DataFrame(
            [
                _fit_department(group)
                for _, group in weekly.groupby(["store_id", "dept_id"], sort=True)
            ]
        )
        selected_counts = (
            selected.groupby(["store_id", "dept_id"])
            .size()
            .rename("selected_series")
            .reset_index()
        )
        store_results = store_results.merge(
            selected_counts,
            on=["store_id", "dept_id"],
            validate="one_to_one",
        )
        store_results.to_csv(
            output_dir / "pooled_price_elasticities_by_store.csv",
            index=False,
        )

    audit = {
        "input": str(input_path),
        "models": {
            "feols": (
                "log(units_sold) ~ log(price) | item-store FE + store-week FE; "
                "positive-sales weeks only"
            ),
            "ppml": (
                "units_sold ~ log(price) | item-store FE + store-week FE; "
                "all weeks including zero-sales weeks"
            ),
        },
        "standard_errors": "CRV1 clustered by item-store",
        "singleton_fixed_effects": (
            "PyFixest drops singleton fixed-effect observations during estimation."
        ),
        "zero_sales": (
            "FeOLS excludes zero-sales weeks because log(units_sold) is undefined; "
            "PPML retains them."
        ),
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
        "dml_source": str(dml_source) if not dml_results.empty else None,
        "item_store_recommendations": (
            output_dir / "item_store_pricing_defaults.csv"
        ).name,
        "by_store_results": (
            None if store_results is None else store_results.to_dict(orient="records")
        ),
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
    parser.add_argument(
        "--by-store",
        action="store_true",
        help="Also estimate department models separately for each store.",
    )
    parser.add_argument(
        "--dml-results",
        type=Path,
        help="Optional per-item-store DML CSV; defaults to Module 2 output if present.",
    )
    args = parser.parse_args()

    results = run_pooled_elasticity(
        args.input,
        args.output_dir,
        args.max_series,
        by_store=args.by_store,
        dml_results_path=args.dml_results,
    )
    print(results.to_string(index=False))
    if args.by_store:
        print("\nDepartment estimates by store:")
        store_results = pd.read_csv(
            args.output_dir / "pooled_price_elasticities_by_store.csv"
        )
        print(store_results.to_string(index=False))
    print(f"\nSaved results and audit to {args.output_dir}")


if __name__ == "__main__":
    main()
