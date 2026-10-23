"""Hold-out accuracy check for weekly unit forecasts on the Module 2 series.

Each item-store series is split in time: everything before the last `holdout_weeks`
weeks is for fitting, and the held-out weeks are predicted one week ahead. Two
forecasts are compared on the same held-out weeks:

- naive: the mean of the last four training weeks (no model, no price);
- regression: ridge regression on the last four weeks of sales plus price, SNAP,
  event, promotion flag and annual seasonality, fitted on the training weeks only.

The score is WAPE (sum of absolute errors divided by total actual units), which is
robust to low-volume series. Lower is better. This measures forecasting accuracy on
history only; it says nothing about whether a price change causes the change in sales.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from retail_engine.module2 import WEEKLY_SQL, _weekly_control_matrix

LAGS = 4
DEFAULT_HOLDOUT_WEEKS = 13


def _features(weekly: pd.DataFrame) -> np.ndarray:
    """Lagged units, price and calendar controls for each week (first LAGS rows padded)."""
    units = weekly["units_sold"].to_numpy(dtype=np.float64)
    lags = np.column_stack(
        [np.concatenate([np.full(k, units[0]), units[:-k]]) for k in range(1, LAGS + 1)]
    )
    controls = _weekly_control_matrix(weekly)
    price = np.log(weekly["price"].to_numpy(dtype=np.float64))[:, None]
    return np.hstack([np.log1p(lags), price, controls])


def forecast_series(weekly: pd.DataFrame, holdout_weeks: int) -> pd.DataFrame:
    """Return actual and predicted units for the held-out weeks of one series."""
    weekly = weekly.sort_values("week_start", kind="stable").reset_index(drop=True)
    split = len(weekly) - holdout_weeks
    if split <= LAGS + 10:
        return pd.DataFrame()
    units = weekly["units_sold"].to_numpy(dtype=np.float64)
    features = _features(weekly)
    target = np.log1p(units)

    model = Ridge(alpha=1.0).fit(features[:split], target[:split])
    regression = np.expm1(model.predict(features[split:]))
    regression = np.clip(regression, 0, None)

    naive = np.full(holdout_weeks, units[split - LAGS : split].mean())
    return pd.DataFrame(
        {
            "item_id": weekly["item_id"].iloc[0],
            "store_id": weekly["store_id"].iloc[0],
            "week_start": weekly["week_start"].iloc[split:].to_numpy(),
            "actual_units": units[split:],
            "naive_forecast": naive,
            "regression_forecast": regression,
        }
    )


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    total = np.sum(actual)
    return float(np.sum(np.abs(actual - forecast)) / total) if total > 0 else float("nan")


def summarise(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, column in (("naive", "naive_forecast"), ("regression", "regression_forecast")):
        rows.append(
            {
                "model": name,
                "series": predictions[["item_id", "store_id"]].drop_duplicates().shape[0],
                "held_out_weeks": len(predictions),
                "wape": wape(predictions["actual_units"].to_numpy(), predictions[column].to_numpy()),
                "mae_units_per_week": float(
                    np.mean(np.abs(predictions["actual_units"] - predictions[column]))
                ),
            }
        )
    return pd.DataFrame(rows)


def run_holdout(
    weekly: pd.DataFrame, holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [
        forecast_series(group, holdout_weeks)
        for _, group in weekly.groupby(["item_id", "store_id"], sort=True)
    ]
    predictions = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return predictions, summarise(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/clean_retail_data.parquet")
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=Path("data/processed/module2/price_elasticities_selected_combinations.csv"),
    )
    parser.add_argument("--holdout-weeks", type=int, default=DEFAULT_HOLDOUT_WEEKS)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/forecast_holdout")
    )
    args = parser.parse_args()

    shortlisted = pd.read_csv(args.series)[["item_id", "store_id"]]
    connection = duckdb.connect()
    connection.register("shortlisted", shortlisted)
    weekly = connection.execute(WEEKLY_SQL, [str(args.input)]).df()
    predictions, summary = run_holdout(weekly, args.holdout_weeks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "holdout_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


def run_pooled_holdout(
    weekly: pd.DataFrame, holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS
) -> pd.DataFrame:
    """One ridge model fitted across all series, after removing each series' own level.

    Each series' training-period mean of log1p(units) is subtracted before fitting, so
    the shared model learns price and calendar responses, not which series sell more.
    The level is added back for prediction. Only training-period values are used.
    """
    feature_blocks, target_blocks, meta = [], [], []
    prepared = {}
    for key, group in weekly.groupby(["item_id", "store_id"], sort=True):
        group = group.sort_values("week_start", kind="stable").reset_index(drop=True)
        split = len(group) - holdout_weeks
        if split <= LAGS + 10:
            continue
        features = _features(group)
        target = np.log1p(group["units_sold"].to_numpy(dtype=np.float64))
        level = target[:split].mean()
        price_level = features[:split, LAGS].mean()
        # Centre lagged sales and log price on this series' training level so every
        # input and the target live on the same scale; calendar flags are unchanged.
        centred = features.copy()
        centred[:, :LAGS] -= level
        centred[:, LAGS] -= price_level
        prepared[key] = (group, centred, target, split, level)
        feature_blocks.append(centred[LAGS:split])
        target_blocks.append(target[LAGS:split] - level)
    model = Ridge(alpha=1.0).fit(np.vstack(feature_blocks), np.concatenate(target_blocks))

    rows = []
    for key, (group, features, target, split, level) in prepared.items():
        units = group["units_sold"].to_numpy(dtype=np.float64)
        pooled = np.clip(np.expm1(model.predict(features[split:]) + level), 0, None)
        naive = np.full(holdout_weeks, units[split - LAGS : split].mean())
        rows.append(
            pd.DataFrame(
                {
                    "item_id": key[0],
                    "store_id": key[1],
                    "week_start": group["week_start"].iloc[split:].to_numpy(),
                    "actual_units": units[split:],
                    "naive_forecast": naive,
                    "pooled_forecast": pooled,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def per_series_comparison(predictions: pd.DataFrame, column: str) -> dict[str, float]:
    per = predictions.groupby(["item_id", "store_id"]).apply(
        lambda d: pd.Series(
            {
                "naive": np.abs(d.actual_units - d.naive_forecast).sum() / d.actual_units.sum(),
                "model": np.abs(d.actual_units - d[column]).sum() / d.actual_units.sum(),
            }
        ),
        include_groups=False,
    )
    return {
        "overall_wape_naive": wape(predictions["actual_units"].to_numpy(), predictions["naive_forecast"].to_numpy()),
        "overall_wape_model": wape(predictions["actual_units"].to_numpy(), predictions[column].to_numpy()),
        "median_wape_naive": float(per["naive"].median()),
        "median_wape_model": float(per["model"].median()),
        "series_model_better": int((per["model"] < per["naive"]).sum()),
        "series_total": int(len(per)),
    }
