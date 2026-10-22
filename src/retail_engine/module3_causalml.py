"""Fit an observational CausalML T-learner diagnostic for inferred promotions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from retail_engine.module2 import WEEK_START_SQL

DEFAULT_MAX_SERIES = 1_000
MIN_TREATMENT_ROWS = 50
RANDOM_STATE = 123
FEATURE_COLUMNS = [
    "lag_1_units",
    "lag_4_units",
    "prior_4_week_mean",
    "snap_share",
    "event_week",
    "annual_sin",
    "annual_cos",
]
OUTPUT_COLUMNS = [
    "cat_id",
    "store_id",
    "evaluation_start",
    "evaluation_end",
    "series_count",
    "evaluation_rows",
    "promotion_rows",
    "promotion_share",
    "observed_promotion_mean_units",
    "observed_control_mean_units",
    "mean_predicted_treatment_effect_units",
    "median_predicted_treatment_effect_units",
    "treatment_effect_p10_units",
    "treatment_effect_p90_units",
    "interpretation",
]
EFFECT_COLUMNS = [
    "series_count",
    "evaluation_rows",
    "promotion_rows",
    "observed_promotion_mean_units",
    "observed_control_mean_units",
    "mean_predicted_treatment_effect_units",
    "median_predicted_treatment_effect_units",
    "interpretation",
]
ITEM_STORE_COLUMNS = ["item_id", "store_id", "cat_id", *EFFECT_COLUMNS]
PRODUCT_COLUMNS = ["item_id", "cat_id", *EFFECT_COLUMNS]
INTERPRETATION = (
    "Observational CATE diagnostic only; inferred price-drop exposure is not "
    "randomized and estimates are not causal effects or recommendations."
)


def _weekly_panel(parquet_path: Path, max_series: int) -> pd.DataFrame:
    if max_series < 1:
        raise ValueError("max_series must be positive")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Module 1 Parquet file not found: {parquet_path}")

    with duckdb.connect() as connection:
        selected_series = connection.execute(
            """
            SELECT item_id, store_id
            FROM read_parquet(?)
            GROUP BY item_id, store_id
            HAVING MAX(promo_flag) = 1
            ORDER BY SUM(units_sold) DESC, item_id, store_id
            LIMIT ?
            """,
            [str(parquet_path), max_series],
        ).df()
        if selected_series.empty:
            raise ValueError("No inferred promotion series were found in the input")

        connection.register("selected_promotion_series", selected_series)
        return connection.execute(
            f"""
            WITH weekly AS (
                SELECT d.item_id,
                       d.store_id,
                       MAX(d.cat_id) AS cat_id,
                       {WEEK_START_SQL} AS week_start,
                       SUM(d.units_sold)::DOUBLE AS units_sold,
                       AVG(d.snap_flag)::DOUBLE AS snap_share,
                       MAX(CASE WHEN d.event_name IS NOT NULL
                                     AND d.event_name <> ''
                                THEN 1 ELSE 0 END)::DOUBLE AS event_week,
                       MAX(d.promo_flag)::INTEGER AS treatment
                FROM read_parquet(?) AS d
                INNER JOIN selected_promotion_series AS s
                    ON d.item_id = s.item_id AND d.store_id = s.store_id
                GROUP BY d.item_id, d.store_id, week_start
            )
            SELECT *,
                   LAG(units_sold, 1) OVER series_window AS lag_1_units,
                   LAG(units_sold, 4) OVER series_window AS lag_4_units,
                   AVG(units_sold) OVER (
                       PARTITION BY item_id, store_id
                       ORDER BY week_start
                       ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                   ) AS prior_4_week_mean
            FROM weekly
            WINDOW series_window AS (
                PARTITION BY item_id, store_id ORDER BY week_start
            )
            ORDER BY item_id, store_id, week_start
            """,
            [str(parquet_path)],
        ).df()


def _prepare_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        raise ValueError("Promotion panel is empty")

    data = panel.copy()
    data["week_start"] = pd.to_datetime(data["week_start"], errors="raise")
    iso_week = data["week_start"].dt.isocalendar().week.astype(float)
    data["annual_sin"] = np.sin(2 * np.pi * iso_week / 52.18)
    data["annual_cos"] = np.cos(2 * np.pi * iso_week / 52.18)
    data = data.dropna(subset=[*FEATURE_COLUMNS, "units_sold", "treatment"])
    if data.empty:
        raise ValueError("No rows remain after constructing lagged pre-treatment controls")

    category_store = pd.get_dummies(
        data[["cat_id", "store_id"]].astype("category"),
        dtype=float,
    )
    features = pd.concat(
        [data[FEATURE_COLUMNS].astype(float), category_store],
        axis=1,
    )
    data = data.reset_index(drop=True)
    features = features.reset_index(drop=True)
    return data, features


def _fit_predict(
    panel: pd.DataFrame,
    *,
    min_treatment_rows: int = MIN_TREATMENT_ROWS,
) -> pd.DataFrame:
    try:
        from causalml.inference.meta import BaseTRegressor
    except ImportError as error:
        raise ImportError(
            "The promotion T-learner requires CausalML; install the optional "
            'dependency with `python -m pip install -e ".[uplift]"`.'
        ) from error

    data, features = _prepare_panel(panel)
    cutoff = data["week_start"].quantile(0.8)
    train_mask = data["week_start"].lt(cutoff).to_numpy()
    test_mask = ~train_mask
    treatment = data["treatment"].to_numpy(dtype=np.int8)
    outcome = data["units_sold"].to_numpy(dtype=float)
    train_treatment = treatment[train_mask]
    test_treatment = treatment[test_mask]
    if (train_treatment == 1).sum() < min_treatment_rows:
        raise ValueError(
            "Insufficient inferred-promotion weeks before the temporal holdout: "
            f"need {min_treatment_rows}, found {(train_treatment == 1).sum()}"
        )
    if (train_treatment == 0).sum() < min_treatment_rows:
        raise ValueError("Insufficient control weeks before the temporal holdout")
    if (test_treatment == 1).sum() < min_treatment_rows:
        raise ValueError(
            "Insufficient inferred-promotion weeks in the temporal holdout: "
            f"need {min_treatment_rows}, found {(test_treatment == 1).sum()}"
        )
    if (test_treatment == 0).sum() < min_treatment_rows:
        raise ValueError("Insufficient control weeks in the temporal holdout")

    estimator = BaseTRegressor(
        learner=RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=50,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        control_name=0,
    )
    estimator.fit(
        X=features.loc[train_mask].to_numpy(dtype=float),
        treatment=train_treatment,
        y=outcome[train_mask],
    )
    treatment_effect = estimator.predict(
        X=features.loc[test_mask].to_numpy(dtype=float),
        treatment=test_treatment,
        verbose=False,
    )
    if treatment_effect.ndim != 2 or treatment_effect.shape[1] != 1:
        raise ValueError(
            "Expected one binary-treatment effect column from CausalML; got "
            f"shape {treatment_effect.shape}"
        )

    evaluated = data.loc[
        test_mask,
        ["item_id", "cat_id", "store_id", "week_start", "treatment", "units_sold"],
    ].copy()
    evaluated["predicted_treatment_effect_units"] = treatment_effect[:, 0]
    return evaluated


def summarise_predicted_effects(
    evaluated: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Aggregate row-level predicted effects (item-store-weeks) to a coarser level.

    Rows are the chronological-holdout weeks. `mean_predicted_treatment_effect_units`
    is the average predicted extra units per item-store-week if promoted, taken over
    all holdout weeks of the group (not only promoted ones), so it describes expected
    responsiveness. The observed promoted/control means are raw holdout averages and
    are NaN when a group has no promoted (or no control) weeks.
    """
    frame = evaluated.copy()
    frame["series_key"] = frame["item_id"].astype(str) + "|" + frame["store_id"].astype(str)
    grouped = frame.groupby(group_columns, as_index=False, sort=True)
    result = grouped.agg(
        series_count=("series_key", "nunique"),
        evaluation_rows=("treatment", "size"),
        promotion_rows=("treatment", "sum"),
        mean_predicted_treatment_effect_units=(
            "predicted_treatment_effect_units",
            "mean",
        ),
        median_predicted_treatment_effect_units=(
            "predicted_treatment_effect_units",
            "median",
        ),
    )
    promoted = (
        frame.loc[frame["treatment"].eq(1)]
        .groupby(group_columns)["units_sold"]
        .mean()
        .rename("observed_promotion_mean_units")
    )
    control = (
        frame.loc[frame["treatment"].eq(0)]
        .groupby(group_columns)["units_sold"]
        .mean()
        .rename("observed_control_mean_units")
    )
    result = result.merge(promoted, on=group_columns, how="left").merge(
        control, on=group_columns, how="left"
    )
    result["interpretation"] = INTERPRETATION
    columns = [*group_columns, *EFFECT_COLUMNS]
    return result[columns].sort_values(
        ["mean_predicted_treatment_effect_units", *group_columns],
        ascending=[False, *([True] * len(group_columns))],
        kind="stable",
    ).reset_index(drop=True)


def estimate_observational_promotion_effects(
    parquet_path: Path,
    output_dir: Path,
    *,
    max_series: int = DEFAULT_MAX_SERIES,
) -> pd.DataFrame:
    panel = _weekly_panel(parquet_path, max_series)
    evaluated = _fit_predict(panel)
    group_columns = ["cat_id", "store_id"]
    results = evaluated.groupby(
        group_columns, as_index=False, sort=True
    ).agg(
        evaluation_start=("week_start", "min"),
        evaluation_end=("week_start", "max"),
        evaluation_rows=("treatment", "size"),
        promotion_rows=("treatment", "sum"),
        mean_predicted_treatment_effect_units=(
            "predicted_treatment_effect_units",
            "mean",
        ),
        median_predicted_treatment_effect_units=(
            "predicted_treatment_effect_units",
            "median",
        ),
        treatment_effect_p10_units=(
            "predicted_treatment_effect_units",
            lambda values: float(values.quantile(0.10)),
        ),
        treatment_effect_p90_units=(
            "predicted_treatment_effect_units",
            lambda values: float(values.quantile(0.90)),
        ),
    )
    promoted_means = (
        evaluated.loc[evaluated["treatment"].eq(1)]
        .groupby(group_columns)["units_sold"]
        .mean()
        .rename("observed_promotion_mean_units")
    )
    control_means = (
        evaluated.loc[evaluated["treatment"].eq(0)]
        .groupby(group_columns)["units_sold"]
        .mean()
        .rename("observed_control_mean_units")
    )
    results = results.merge(
        promoted_means,
        on=group_columns,
        how="left",
        validate="one_to_one",
    ).merge(
        control_means,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )
    series_counts = (
        panel[["cat_id", "store_id", "item_id"]]
        .drop_duplicates()
        .groupby(["cat_id", "store_id"], as_index=False)
        .size()
        .rename(columns={"size": "series_count"})
    )
    results = results.merge(series_counts, on=["cat_id", "store_id"], validate="one_to_one")
    results["promotion_share"] = results["promotion_rows"] / results["evaluation_rows"]
    results["interpretation"] = INTERPRETATION
    results["evaluation_start"] = results["evaluation_start"].dt.date.astype(str)
    results["evaluation_end"] = results["evaluation_end"].dt.date.astype(str)
    results = results[OUTPUT_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "promotion_tlearner_observational.csv", index=False)
    item_store = summarise_predicted_effects(evaluated, ["item_id", "store_id", "cat_id"])
    product = summarise_predicted_effects(evaluated, ["item_id", "cat_id"])
    item_store[ITEM_STORE_COLUMNS].to_csv(
        output_dir / "promotion_tlearner_item_store.csv", index=False
    )
    product[PRODUCT_COLUMNS].to_csv(
        output_dir / "promotion_tlearner_product.csv", index=False
    )
    audit = {
        "source_file": str(parquet_path),
        "method": "CausalML BaseTRegressor T-learner with random-forest outcome models",
        "target": "Weekly units_sold; predicted treatment effects are in units per item-store-week.",
        "treatment": (
            "Module 1 inferred promo_flag. M5 has no retailer-provided promotion "
            "assignment; price-drop exposure is observational."
        ),
        "controls": [
            "one-week lagged units",
            "four-week lagged units",
            "mean units over the preceding four weeks",
            "SNAP share",
            "event-week indicator",
            "annual sine/cosine seasonality",
            "store and department indicators",
        ],
        "holdout": "Chronological: earliest 80% of weeks for fitting; latest 20% for evaluation.",
        "selected_series_limit": max_series,
        "selected_series_count": int(
            panel[["item_id", "store_id"]].drop_duplicates().shape[0]
        ),
        "training_rows": int(
            panel["week_start"].lt(panel["week_start"].quantile(0.8)).sum()
        ),
        "evaluation_rows": int(len(evaluated)),
        "minimum_arm_rows": MIN_TREATMENT_ROWS,
        "random_state": RANDOM_STATE,
        "causal_identification": (
            "Not established. Promotion assignment is inferred and non-random; "
            "unmeasured confounding, stock availability, and measurement errors "
            "remain. Do not interpret as causal uplift or use for decisions."
        ),
        "output_file": "promotion_tlearner_observational.csv",
        "item_store_output_file": "promotion_tlearner_item_store.csv",
        "product_output_file": "promotion_tlearner_product.csv",
        "product_level_note": (
            "Item-store rows give store-by-product predictions; product rows pool "
            "the same item across the selected stores. With a chronological holdout "
            "of about one year, per-item estimates are noisy and only the ranking "
            "within the selected series should be read, not the magnitudes."
        ),
    }
    (output_dir / "promotion_tlearner_observational_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
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
        default=Path("data/processed/module3_causalml"),
        help="Directory for the observational CATE table and audit JSON",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=DEFAULT_MAX_SERIES,
        help="Maximum promoted item-store series, ranked by total sales",
    )
    args = parser.parse_args()
    results = estimate_observational_promotion_effects(
        args.parquet,
        args.output_dir,
        max_series=args.max_series,
    )
    print(f"Department/store observational estimates: {len(results)}")
    print(f"Item-store estimates: {args.output_dir / 'promotion_tlearner_item_store.csv'}")
    print(f"Product estimates: {args.output_dir / 'promotion_tlearner_product.csv'}")
    print(f"Results: {args.output_dir / 'promotion_tlearner_observational.csv'}")
    print(f"Audit: {args.output_dir / 'promotion_tlearner_observational_audit.json'}")


if __name__ == "__main__":
    main()
