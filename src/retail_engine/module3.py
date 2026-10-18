"""Estimate promotion uplift and contribution impact from the Module 1 table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from retail_engine.module2 import WEEK_START_SQL

CATEGORY_MARGIN_RATES = {"FOODS": 0.25, "HOUSEHOLD": 0.35, "HOBBIES": 0.40}
BASELINE_WEEKS = 4
BASELINE_LOOKBACK_WEEKS = 8
POST_PROMO_WEEKS = 2
OUTPUT_COLUMNS = [
    "item_id",
    "store_id",
    "cat_id",
    "avg_discount_depth",
    "margin_rate_used",
    "promotion_start",
    "promotion_end",
    "promotion_weeks",
    "price_persists",
    "baseline_weeks_used",
    "baseline_units_per_week",
    "baseline_sales",
    "promo_sales",
    "incremental_units",
    "baseline_revenue",
    "promo_revenue",
    "incremental_revenue",
    "baseline_profit",
    "promo_profit",
    "incremental_margin",
    "discount_cost",
    "promotion_ROI",
    "post_promo_weeks_observed",
    "post_promo_sales",
    "post_promo_dip_units",
    "recommendation",
    "analysis_status",
]


def _validate_gross_margin_rate(gross_margin_rate: float) -> None:
    if not 0 < gross_margin_rate < 1:
        raise ValueError("gross_margin_rate must be greater than 0 and less than 1")


def _promotion_runs(weekly: pd.DataFrame) -> list[pd.DataFrame]:
    promoted = weekly.loc[weekly["promo_flag"].eq(1)].copy()
    if promoted.empty:
        return []

    promoted = promoted.sort_values("week_start").reset_index(drop=True)
    starts_new_run = promoted["week_start"].diff().dt.days.ne(7)
    run_ids = starts_new_run.cumsum()
    return [
        run.reset_index(drop=True)
        for _, run in promoted.groupby(run_ids, sort=False)
    ]


def _evaluate_series_promotions(
    weekly: pd.DataFrame,
    gross_margin_rate: float | None = None,
    baseline_weeks: int = BASELINE_WEEKS,
    baseline_lookback_weeks: int = BASELINE_LOOKBACK_WEEKS,
    post_promo_weeks: int = POST_PROMO_WEEKS,
) -> list[dict[str, object]]:
    if baseline_weeks <= 0 or baseline_lookback_weeks < baseline_weeks:
        raise ValueError(
            "baseline_lookback_weeks must be at least baseline_weeks, both positive"
        )
    if post_promo_weeks < 0:
        raise ValueError("post_promo_weeks cannot be negative")

    ordered = weekly.sort_values("week_start").reset_index(drop=True).copy()
    ordered["week_start"] = pd.to_datetime(ordered["week_start"], errors="raise")
    if ordered["week_start"].duplicated().any():
        raise ValueError("weekly input must contain at most one row per week")

    results: list[dict[str, object]] = []
    for episode in _promotion_runs(ordered):
        cat_id = str(episode["cat_id"].iloc[0])
        rate = (
            gross_margin_rate
            if gross_margin_rate is not None
            else CATEGORY_MARGIN_RATES[cat_id]
        )
        _validate_gross_margin_rate(rate)
        first_week = episode["week_start"].iloc[0]
        last_week = episode["week_start"].iloc[-1]
        nxt = ordered.loc[
            ordered["week_start"] == last_week + pd.Timedelta(weeks=1), "price"
        ]
        price_persists = bool(
            len(nxt) and nxt.iloc[0] <= episode["price"].iloc[-1] * 1.05
        )
        duration = len(episode)
        promo_sales = float(episode["units_sold"].sum())
        baseline_history = ordered.loc[
            ordered["week_start"].lt(first_week)
            & ordered["week_start"].ge(
                first_week - pd.Timedelta(weeks=baseline_lookback_weeks)
            )
            & ordered["promo_flag"].eq(0)
        ].tail(baseline_weeks)
        enough_history = len(baseline_history) == baseline_weeks

        base: dict[str, object] = {
            "item_id": str(episode["item_id"].iloc[0]),
            "store_id": str(episode["store_id"].iloc[0]),
            "cat_id": cat_id,
            "avg_discount_depth": float(episode["discount_depth"].mean()),
            "margin_rate_used": rate,
            "promotion_start": first_week.date().isoformat(),
            "promotion_end": (last_week + pd.Timedelta(days=6)).date().isoformat(),
            "promotion_weeks": duration,
            "price_persists": price_persists,
            "baseline_weeks_used": len(baseline_history),
            "baseline_units_per_week": None,
            "baseline_sales": None,
            "promo_sales": promo_sales,
            "incremental_units": None,
            "baseline_revenue": None,
            "promo_revenue": float((episode["price"] * episode["units_sold"]).sum()),
            "incremental_revenue": None,
            "baseline_profit": None,
            "promo_profit": None,
            "incremental_margin": None,
            "discount_cost": float(
                (
                    (episode["regular_price"] - episode["price"]).clip(lower=0)
                    * episode["units_sold"]
                ).sum()
            ),
            "promotion_ROI": None,
            "post_promo_weeks_observed": 0,
            "post_promo_sales": None,
            "post_promo_dip_units": None,
            "recommendation": (
                "Possible permanent price change - review"
                if price_persists
                else "Review - insufficient baseline"
            ),
            "analysis_status": "insufficient_baseline",
        }
        if not enough_history:
            results.append(base)
            continue

        baseline_units_per_week = float(baseline_history["units_sold"].mean())
        baseline_sales = baseline_units_per_week * duration
        baseline_revenue = float(
            baseline_units_per_week * episode["regular_price"].sum()
        )
        baseline_profit = baseline_revenue * rate
        unit_cost = episode["regular_price"] * (1 - rate)
        promo_profit = float(
            ((episode["price"] - unit_cost) * episode["units_sold"]).sum()
        )
        incremental_units = promo_sales - baseline_sales
        incremental_revenue = float(base["promo_revenue"]) - baseline_revenue
        incremental_margin = promo_profit - baseline_profit
        discount_cost = float(base["discount_cost"])
        if incremental_units <= 0:
            recommendation = "Stop promotion"
        elif incremental_margin <= 0:
            recommendation = "Reduce discount"
        else:
            recommendation = "Continue promotion"
        if price_persists:
            recommendation = "Possible permanent price change - review"

        post_sales: float | None = None
        post_dip: float | None = None
        observed_post_weeks = 0
        if post_promo_weeks:
            expected_post_starts = pd.date_range(
                last_week + pd.Timedelta(weeks=1),
                periods=post_promo_weeks,
                freq="7D",
            )
            post = ordered.loc[
                ordered["week_start"].isin(expected_post_starts)
            ]
            observed_post_weeks = len(post)
            if observed_post_weeks == post_promo_weeks:
                post_sales = float(post["units_sold"].sum())
                post_dip = post_sales - baseline_units_per_week * post_promo_weeks

        base.update(
            {
                "baseline_units_per_week": baseline_units_per_week,
                "baseline_sales": baseline_sales,
                "incremental_units": incremental_units,
                "baseline_revenue": baseline_revenue,
                "incremental_revenue": incremental_revenue,
                "baseline_profit": baseline_profit,
                "promo_profit": promo_profit,
                "incremental_margin": incremental_margin,
                "promotion_ROI": (
                    incremental_margin / discount_cost
                    if discount_cost > 0
                    else None
                ),
                "post_promo_weeks_observed": observed_post_weeks,
                "post_promo_sales": post_sales,
                "post_promo_dip_units": post_dip,
                "recommendation": recommendation,
                "analysis_status": "complete",
            }
        )
        results.append(base)

    return results


def _weekly_promotion_data(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Module 1 Parquet file not found: {parquet_path}")

    with duckdb.connect() as connection:
        promoted_series = connection.execute(
            """
            SELECT DISTINCT item_id, store_id
            FROM read_parquet(?)
            WHERE promo_flag = 1
            ORDER BY item_id, store_id
            """,
            [str(parquet_path)],
        ).df()
        if promoted_series.empty:
            return pd.DataFrame(
                columns=[
                    "item_id",
                    "store_id",
                    "cat_id",
                    "week_start",
                    "units_sold",
                    "price",
                    "regular_price",
                    "promo_flag",
                    "discount_depth",
                ]
            )

        connection.register("promoted_series", promoted_series)
        return connection.execute(
            f"""
            SELECT d.item_id, d.store_id, MAX(d.cat_id) AS cat_id,
                   {WEEK_START_SQL} AS week_start,
                   SUM(d.units_sold)::DOUBLE AS units_sold,
                   AVG(d.price)::DOUBLE AS price,
                   AVG(d.regular_price)::DOUBLE AS regular_price,
                   MAX(d.promo_flag)::INTEGER AS promo_flag,
                   AVG(d.discount_depth)::DOUBLE AS discount_depth
            FROM read_parquet(?) AS d
            INNER JOIN promoted_series AS p
                ON d.item_id = p.item_id AND d.store_id = p.store_id
            GROUP BY d.item_id, d.store_id, week_start
            ORDER BY d.item_id, d.store_id, week_start
            """,
            [str(parquet_path)],
        ).df()


def _audit_summary(
    results: pd.DataFrame,
    parquet_path: Path,
    gross_margin_rate: float | None,
) -> dict[str, object]:
    recommendation_counts = {
        str(key): int(value)
        for key, value in results["recommendation"]
        .value_counts()
        .sort_index()
        .items()
    } if not results.empty else {}
    status_counts = {
        str(key): int(value)
        for key, value in results["analysis_status"]
        .value_counts()
        .sort_index()
        .items()
    } if not results.empty else {}
    return {
        "source_file": str(parquet_path),
        "promotion_definition": (
            "A promotion window is one or more consecutive Saturday-start M5 "
            "weeks with promo_flag=1 in the Module 1 inferred price-drop proxy."
        ),
        "persistent_price_review_rule": (
            "If the next observed week price is at most 105% of the final "
            "promotion-week price, recommend Possible permanent price change - "
            "review instead of Continue/Reduce/Stop. If baseline history is "
            "insufficient, the persistence review takes precedence."
        ),
        "baseline_method": (
            f"Mean weekly units from the {BASELINE_WEEKS} latest non-promotion "
            f"weeks in the {BASELINE_LOOKBACK_WEEKS} calendar weeks before the "
            "promotion; require all baseline weeks or mark insufficient."
        ),
        "category_margin_rate_assumptions": CATEGORY_MARGIN_RATES,
        "gross_margin_rate_override": gross_margin_rate,
        "unit_cost_method": (
            "Estimated unit cost is regular_price * (1 - margin_rate_used); "
            "cost is held fixed during the promotion."
        ),
        "promotion_roi_method": (
            "Incremental margin divided by discount cost "
            "((regular_price - promo price) * promoted units)."
        ),
        "post_promotion_check": (
            f"Compare the {POST_PROMO_WEEKS} following calendar weeks with "
            "baseline weekly units; only report a dip when all weeks are present."
        ),
        "promotions_evaluated": len(results),
        "analysis_status_counts": status_counts,
        "recommendation_counts": recommendation_counts,
        "output_file": "promotion_uplift.csv",
    }


def build_promotion_uplift(
    parquet_path: Path,
    output_dir: Path,
    gross_margin_rate: float | None = None,
) -> pd.DataFrame:
    if gross_margin_rate is not None:
        _validate_gross_margin_rate(gross_margin_rate)
    weekly = _weekly_promotion_data(parquet_path)
    promotions: list[dict[str, object]] = []
    if not weekly.empty:
        for _, series in weekly.groupby(["item_id", "store_id"], sort=False):
            promotions.extend(
                _evaluate_series_promotions(
                    series,
                    gross_margin_rate=gross_margin_rate,
                )
            )

    results = pd.DataFrame(promotions, columns=OUTPUT_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "promotion_uplift.csv"
    audit_path = output_dir / "promotion_uplift_audit.json"
    results.to_csv(output_path, index=False)
    audit = _audit_summary(results, parquet_path, gross_margin_rate)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
        help="Module 1 clean retail Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/module3"),
        help="Directory for the uplift table and audit JSON",
    )
    parser.add_argument(
        "--gross-margin-rate",
        type=float,
        default=None,
        help="Override the assumed category-specific regular-price gross margins",
    )
    args = parser.parse_args()
    results = build_promotion_uplift(
        args.parquet,
        args.output_dir,
        gross_margin_rate=args.gross_margin_rate,
    )
    print(f"Promotion episodes evaluated: {len(results)}")
    if not results.empty:
        print(results["recommendation"].value_counts().to_string())
    print(f"Results: {args.output_dir / 'promotion_uplift.csv'}")
    print(f"Audit: {args.output_dir / 'promotion_uplift_audit.json'}")


if __name__ == "__main__":
    main()
