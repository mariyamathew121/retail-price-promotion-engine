"""Build the 'Proof and What-if' page: a standalone HTML file with embedded results.

Two parts, both labelled as scenarios:

- Forecast scoreboard: the pooled hold-out forecast against the naive forecast over
  13-, 26- and 52-week test windows (from forecast_holdout.py).
- What-if simulator: for each of the 100 Module 2 series, the per-series elasticity
  with its 95% interval, applied to the series' recent average units and price using
  the constant-elasticity form: units scale by (1 + price change) ** elasticity.

The page does not claim that a price change causes the predicted sales change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from retail_engine.forecast_holdout import per_series_comparison, run_pooled_holdout
from retail_engine.module2 import WEEKLY_SQL

HOLDOUT_WINDOWS = (13, 26, 52)
BASELINE_WEEKS = 13
PRICE_CHANGE_RANGE = (-0.20, 0.20)


def simulate_change(
    base_units: float, base_price: float, elasticity: float, price_change: float
) -> dict[str, float]:
    """Predicted weekly units and revenue after a relative price change.

    Constant-elasticity scenario: units scale by (1 + change) ** elasticity, so the
    revenue index is (1 + change) ** (1 + elasticity). Negative changes are price cuts.
    """
    factor = (1.0 + price_change) ** elasticity
    revenue_factor = (1.0 + price_change) ** (1.0 + elasticity)
    return {
        "units": base_units * factor,
        "units_change_pct": (factor - 1.0) * 100.0,
        "revenue_change_pct": (revenue_factor - 1.0) * 100.0,
        "price": base_price * (1.0 + price_change),
    }


def scoreboard(weekly: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for weeks in HOLDOUT_WINDOWS:
        predictions = run_pooled_holdout(weekly, weeks)
        result = per_series_comparison(predictions, "pooled_forecast")
        rows.append(
            {
                "test_weeks": weeks,
                "naive_error": result["overall_wape_naive"],
                "model_error": result["overall_wape_model"],
                "naive_median": result["median_wape_naive"],
                "model_median": result["median_wape_model"],
                "series_won": result["series_model_better"],
                "series_total": result["series_total"],
            }
        )
    return rows


def baseline_table(weekly: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    recent = (
        weekly.sort_values("week_start", kind="stable")
        .groupby(["item_id", "store_id"], sort=True)
        .tail(BASELINE_WEEKS)
        .groupby(["item_id", "store_id"], as_index=False)
        .agg(base_units=("units_sold", "mean"), base_price=("price", "mean"))
    )
    return series.merge(recent, on=["item_id", "store_id"], how="inner")


def build_payload(weekly: pd.DataFrame, elasticities: pd.DataFrame) -> dict[str, object]:
    base = baseline_table(weekly, elasticities)
    series = [
        {
            "id": f"{row.item_id} at {row.store_id}",
            "item_id": row.item_id,
            "store_id": row.store_id,
            "base_units": round(float(row.base_units), 2),
            "base_price": round(float(row.base_price), 2),
            "elasticity": round(float(row.elasticity), 3),
            "ci_lower": round(float(row.ci_lower), 3),
            "ci_upper": round(float(row.ci_upper), 3),
            "demand_type": str(row.demand_type),
        }
        for row in base.itertuples(index=False)
    ]
    return {
        "scoreboard": scoreboard(weekly),
        "series": series,
        "price_change_range": PRICE_CHANGE_RANGE,
        "baseline_weeks": BASELINE_WEEKS,
    }


def load_inputs(parquet: Path, series_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    elasticities = pd.read_csv(series_csv)
    shortlisted = elasticities[["item_id", "store_id"]]
    connection = duckdb.connect()
    connection.register("shortlisted", shortlisted)
    weekly = connection.execute(WEEKLY_SQL, [str(parquet)]).df()
    elasticities = elasticities.loc[elasticities["fit_status"].eq("estimated")]
    return weekly, elasticities


def render_html(payload: dict[str, object], template: str) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return template.replace("/*__DATA__*/null", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=Path("data/processed/clean_retail_data.parquet")
    )
    parser.add_argument(
        "--elasticities",
        type=Path,
        default=Path("data/processed/module2/price_elasticities.csv"),
    )
    parser.add_argument(
        "--template", type=Path, default=Path("outputs/showcase/proof_and_whatif.template.html")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/showcase/proof_and_whatif.html")
    )
    parser.add_argument(
        "--data-output", type=Path, default=Path("data/processed/showcase/showcase.json")
    )
    args = parser.parse_args()

    weekly, elasticities = load_inputs(args.parquet, args.elasticities)
    payload = build_payload(weekly, elasticities)
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    args.data_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = render_html(payload, args.template.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Scoreboard windows: {[r['test_weeks'] for r in payload['scoreboard']]}")
    print(f"Series in simulator: {len(payload['series'])}")
    print(f"Wrote {args.output} and {args.data_output}")


if __name__ == "__main__":
    main()
