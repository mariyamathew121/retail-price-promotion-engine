"""Generate time-series diagnostics for the strongest positive M5 DML cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from retail_engine.module2 import WEEK_START_SQL


def _reason(row: pd.Series, price_change_weeks: int, price_trend_corr: float) -> str:
    if price_change_weeks <= 5:
        return (
            f"Only {price_change_weeks} observed price changes across "
            f"{row['weeks_used']} weeks make the positive estimate sensitive to "
            "sparse price variation; do not interpret it as a demand response."
        )
    if abs(price_trend_corr) >= 0.30:
        return (
            f"Log price is correlated with time (r={price_trend_corr:.2f}) and "
            f"there are only {price_change_weeks} price changes; time-varying "
            "confounding is a plausible explanation for the positive estimate."
        )
    return (
        f"The very large positive estimate ({row['elasticity']:.2f}) persists "
        f"with {price_change_weeks} price changes and {row['price_levels']} price "
        "levels; investigate time-varying confounding, inventory, and price "
        "measurement rather than acting on it."
    )


def generate_cases(
    results_path: Path,
    parquet_path: Path,
    output_dir: Path,
    number_of_cases: int = 5,
) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    positive = results.loc[results["elasticity"] > 0].sort_values(
        ["elasticity", "item_id", "store_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    if len(positive) < 3:
        raise ValueError(
            f"At least three positive DML estimates are required; found {len(positive)}"
        )
    cases = positive.head(min(max(number_of_cases, 3), 5)).copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as connection:
        connection.register("selected_cases", cases[["item_id", "store_id"]])
        weekly = connection.execute(
            f"""
            SELECT d.item_id, d.store_id, {WEEK_START_SQL} AS week_start,
                   SUM(d.units_sold)::DOUBLE AS units_sold,
                   AVG(d.price)::DOUBLE AS price
            FROM read_parquet(?) AS d
            INNER JOIN selected_cases AS s
                ON d.item_id = s.item_id AND d.store_id = s.store_id
            GROUP BY d.item_id, d.store_id, week_start
            ORDER BY d.item_id, d.store_id, week_start
            """,
            [str(parquet_path)],
        ).df()

    cases["price_change_weeks"] = 0
    cases["price_trend_correlation"] = float("nan")
    report_rows: list[str] = []
    for _, row in cases.iterrows():
        series = weekly.loc[
            weekly["item_id"].eq(row["item_id"])
            & weekly["store_id"].eq(row["store_id"])
        ].sort_values("week_start")
        if series.empty:
            raise ValueError(
                f"No weekly source rows found for {row['item_id']} at {row['store_id']}"
            )
        series = series.copy()
        series["week_start"] = pd.to_datetime(series["week_start"])
        log_price = series["price"].to_numpy().astype(float)
        price_change_weeks = int((pd.Series(log_price).diff().abs() > 1e-9).sum())
        price_trend_corr = (
            float(pd.Series(log_price).corr(pd.Series(range(len(log_price)))))
            if log_price.std() > 0
            else float("nan")
        )
        cases.loc[
            cases["item_id"].eq(row["item_id"])
            & cases["store_id"].eq(row["store_id"]),
            "price_change_weeks",
        ] = price_change_weeks
        cases.loc[
            cases["item_id"].eq(row["item_id"])
            & cases["store_id"].eq(row["store_id"]),
            "price_trend_correlation",
        ] = price_trend_corr
        reason = _reason(row, price_change_weeks, price_trend_corr)

        filename = f"positive_{row['item_id']}_{row['store_id']}.png"
        figure, units_axis = plt.subplots(figsize=(11, 4.6))
        price_axis = units_axis.twinx()
        units_line = units_axis.plot(
            series["week_start"],
            series["units_sold"],
            color="#1769AA",
            linewidth=1.35,
            label="Weekly units sold",
        )
        price_line = price_axis.plot(
            series["week_start"],
            series["price"],
            color="#D97732",
            linewidth=1.7,
            label="Weekly average price",
        )
        units_axis.set_xlabel("Week")
        units_axis.set_ylabel("Units sold", color="#1769AA")
        price_axis.set_ylabel("Price", color="#D97732")
        units_axis.tick_params(axis="y", labelcolor="#1769AA")
        price_axis.tick_params(axis="y", labelcolor="#D97732")
        units_axis.grid(axis="y", alpha=0.22)
        units_axis.set_title(
            f"{row['item_id']} / {row['store_id']} — positive DML diagnostic "
            f"(estimate {row['elasticity']:.2f}, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}])"
        )
        lines = units_line + price_line
        units_axis.legend(lines, [line.get_label() for line in lines], loc="upper left")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
        report_rows.append(
            f"| {row['item_id']} / {row['store_id']} | "
            f"{row['elasticity']:.2f} "
            f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}] | "
            f"{price_change_weeks} | {reason} | `{filename}` |"
        )

    report = [
        "# Positive item-store elasticity diagnostics",
        "",
        "The plots show weekly units and average price for the five largest "
        "positive item-store DML estimates among the 100 selected series. "
        "The vertical axes are separate (units and currency price); visual "
        "co-movement is descriptive, not causal evidence. These estimates are "
        "classified Unclear and are not price recommendations.",
        "",
        "| Item / store | DML estimate (95% CI) | Price-change weeks | Diagnostic reason | Plot |",
        "|---|---:|---:|---|---|",
        *report_rows,
        "",
        "A positive estimate—even with an interval excluding zero—is not a "
        "plausible demand curve by itself. Possible explanations include "
        "endogenous pricing, stock availability, common time shocks, sparse "
        "price variation, and the use of `log1p(units_sold)` as the outcome.",
        "",
    ]
    (output_dir / "positive_elasticity_cases.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/processed/module2/price_elasticities.csv"),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--number-of-cases", type=int, default=5)
    args = parser.parse_args()
    cases = generate_cases(
        args.results, args.parquet, args.output_dir, args.number_of_cases
    )
    print(
        cases[
            [
                "item_id",
                "store_id",
                "elasticity",
                "price_change_weeks",
                "price_trend_correlation",
            ]
        ].to_string(index=False)
    )
    print(f"Saved {len(cases)} plots and positive_elasticity_cases.md to {args.output_dir}")


if __name__ == "__main__":
    main()
