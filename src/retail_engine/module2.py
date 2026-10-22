"""Estimate item-store price elasticities from the Module 1 Parquet table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

Z_95 = 1.959963984540054
WEEK_START_SQL = (
    "CAST(date AS DATE) - ((dayofweek(CAST(date AS DATE)) + 1) % 7)::INTEGER"
)
SUMMARY_SQL = f"""
WITH daily AS (
    SELECT item_id, store_id, units_sold, price,
           {WEEK_START_SQL} AS week_start
    FROM read_parquet(?)
), summary AS (
    SELECT item_id, store_id,
           SUM(units_sold)::BIGINT AS total_units,
           COUNT(DISTINCT week_start)::INTEGER AS weeks_observed,
           COUNT(DISTINCT CASE WHEN units_sold > 0 THEN week_start END)::INTEGER
               AS positive_weeks,
           COUNT(DISTINCT price)::INTEGER AS price_levels,
           MIN(price)::DOUBLE AS min_price,
           MAX(price)::DOUBLE AS max_price
    FROM daily
    GROUP BY item_id, store_id
)
SELECT *, (max_price / min_price - 1)::DOUBLE AS price_range_pct
FROM summary
WHERE weeks_observed >= ?
  AND positive_weeks >= ?
  AND price_levels >= ?
  AND max_price / min_price - 1 >= ?
ORDER BY total_units DESC, price_range_pct DESC, item_id, store_id
LIMIT ?
"""
WEEKLY_SQL = f"""
SELECT d.item_id, d.store_id, {WEEK_START_SQL} AS week_start,
       SUM(d.units_sold)::DOUBLE AS units_sold,
       AVG(d.price)::DOUBLE AS price,
       AVG(d.snap_flag)::DOUBLE AS snap_share,
       MAX(CASE WHEN d.event_name IS NOT NULL AND d.event_name <> ''
                THEN 1 ELSE 0 END)::DOUBLE AS event_week,
       MAX(d.promo_flag)::DOUBLE AS promo_week
FROM read_parquet(?) AS d
INNER JOIN shortlisted AS s
    ON d.item_id = s.item_id AND d.store_id = s.store_id
GROUP BY d.item_id, d.store_id, week_start
ORDER BY d.item_id, d.store_id, week_start
"""
RESULT_COLUMNS = [
    "item_id",
    "store_id",
    "elasticity",
    "conf_interval",
    "ci_lower",
    "ci_upper",
    "naive_elasticity",
    "naive_conf_interval",
    "ols_elasticity",
    "ols_conf_interval",
    "demand_type",
    "pricing_recommendation",
    "positive_elasticity_explanation",
    "weeks_used",
    "price_levels",
    "price_range_pct",
    "fit_status",
]


def _weekly_control_matrix(weekly: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(weekly["week_start"])
    angle = 2 * np.pi * dates.dt.dayofyear.to_numpy(dtype=np.float64) / 365.25
    return np.column_stack(
        [
            np.sin(angle),
            np.cos(angle),
            weekly["snap_share"].to_numpy(dtype=np.float64),
            weekly["event_week"].to_numpy(dtype=np.float64),
            weekly["promo_week"].to_numpy(dtype=np.float64),
        ]
    )


def _calendar_week_numbers(week_start: pd.Series) -> np.ndarray:
    """Whole weeks since the first observed week, so gaps keep their distance."""
    days = (pd.to_datetime(week_start) - pd.to_datetime(week_start).min()).dt.days
    if (days % 7).any():
        raise ValueError("week_start values must be 7-day multiples apart")
    return (days // 7).to_numpy(dtype=np.int64)


def _hac_standard_error(
    influence: np.ndarray,
    week_indices: np.ndarray,
    max_lag: int = 4,
) -> float:
    """Newey-West standard error, pairing observations by actual week distance."""
    sample_size = len(influence)
    if sample_size < 2:
        return float("nan")
    if np.any(np.diff(week_indices) <= 0):
        raise ValueError("week_indices must be strictly increasing")
    long_run_variance = float(np.dot(influence, influence) / sample_size)
    for lag in range(1, max_lag + 1):
        right = np.searchsorted(week_indices, week_indices + lag)
        valid = right < sample_size
        left = np.flatnonzero(valid)
        right = right[valid]
        exact_match = week_indices[right] == week_indices[left] + lag
        left = left[exact_match]
        right = right[exact_match]
        if len(left) == 0:
            continue
        weight = 1 - lag / (max_lag + 1)
        covariance_sum = float(np.dot(influence[left], influence[right]) / sample_size)
        long_run_variance += 2 * weight * covariance_sum
    return float(np.sqrt(max(long_run_variance, 0.0) / sample_size))


def _ols_log_log(
    log_price: np.ndarray,
    log_units: np.ndarray,
    controls: np.ndarray,
    week_indices: np.ndarray,
) -> tuple[float, float]:
    nuisance = np.column_stack(
        [np.ones(len(log_price), dtype=np.float64), controls]
    )
    price_projection = nuisance @ np.linalg.lstsq(
        nuisance, log_price, rcond=None
    )[0]
    price_residual = log_price - price_projection
    residual_variance = float(np.mean(price_residual**2))
    if residual_variance < 1e-8:
        raise ValueError("Price is not independently identified by OLS controls")
    estimate = float(np.mean(price_residual * log_units) / residual_variance)
    residuals = log_units - nuisance @ np.linalg.lstsq(
        nuisance, log_units - estimate * log_price, rcond=None
    )[0] - estimate * log_price
    price_influence = price_residual * residuals / residual_variance
    se = _hac_standard_error(price_influence, week_indices)
    return estimate, se


def _crossfit_dml(
    log_price: np.ndarray,
    log_units: np.ndarray,
    controls: np.ndarray,
    week_indices: np.ndarray,
    *,
    folds: int,
    trees: int,
    seed: int,
) -> tuple[float, float]:
    sample_size = len(log_price)
    if sample_size < folds * 5:
        raise ValueError("Too few observations for the requested cross-fitting folds")

    predictions_price = np.empty(sample_size, dtype=np.float64)
    predictions_units = np.empty(sample_size, dtype=np.float64)
    blocks = np.array_split(np.arange(sample_size), folds)
    for fold, test_indices in enumerate(blocks):
        train_indices = np.setdiff1d(
            np.arange(sample_size), test_indices, assume_unique=True
        )
        price_model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=seed + fold,
            n_jobs=1,
        )
        units_model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=seed + folds + fold,
            n_jobs=1,
        )
        price_model.fit(controls[train_indices], log_price[train_indices])
        units_model.fit(controls[train_indices], log_units[train_indices])
        predictions_price[test_indices] = price_model.predict(controls[test_indices])
        predictions_units[test_indices] = units_model.predict(controls[test_indices])

    price_residual = log_price - predictions_price
    units_residual = log_units - predictions_units
    residual_variance = float(np.mean(price_residual**2))
    if residual_variance < 1e-8:
        raise ValueError("No independent price variation remains after controls")

    estimate = float(
        np.mean(price_residual * units_residual) / residual_variance
    )
    influence = (
        price_residual * (units_residual - estimate * price_residual)
        / residual_variance
    )
    standard_error = _hac_standard_error(influence, week_indices)
    return estimate, standard_error


def _interval(estimate: float, standard_error: float) -> tuple[float, float]:
    return (
        estimate - Z_95 * standard_error,
        estimate + Z_95 * standard_error,
    )


def classify_elasticity(
    elasticity: float, lower: float, upper: float
) -> tuple[str, str]:
    """Shared demand-type rule for per-SKU DML and pooled department estimates.

    Price-sensitive only when the whole 95% interval is below -1; less
    sensitive when the interval is below 0 but may reach -1; otherwise
    unclear. A positive estimate, or an interval that includes zero, is
    never acted on. `elasticity` is kept for the caller's signature and does
    not affect the class.
    """
    if upper < -1:
        return "Price-sensitive", "Test a controlled discount or promotion"
    if upper < 0:
        return "Less sensitive", "Protect price or test a small increase"
    return "Unclear", "Flag for review; take no pricing action"


_recommendation = classify_elasticity


def _estimate_combination(
    weekly: pd.DataFrame,
    selection: pd.Series,
    *,
    folds: int,
    trees: int,
) -> dict[str, object]:
    weekly = weekly.sort_values("week_start", kind="stable").reset_index(drop=True)
    modeling_data = weekly.copy().reset_index(drop=True)
    base: dict[str, object] = {
        "item_id": selection["item_id"],
        "store_id": selection["store_id"],
        "elasticity": np.nan,
        "conf_interval": "",
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "naive_elasticity": np.nan,
        "naive_conf_interval": "",
        "ols_elasticity": np.nan,
        "ols_conf_interval": "",
        "demand_type": "Unclear",
        "pricing_recommendation": "No action; review estimation status",
        "positive_elasticity_explanation": "",
        "weeks_used": len(modeling_data),
        "price_levels": int(selection["price_levels"]),
        "price_range_pct": float(selection["price_range_pct"]),
        "fit_status": "not_estimable",
    }
    if len(modeling_data) < folds * 5:
        base["fit_status"] = "not_estimable: too few observed weeks"
        return base

    controls = _weekly_control_matrix(modeling_data)
    log_price = np.log(modeling_data["price"].to_numpy(dtype=np.float64))
    log_units = np.log1p(modeling_data["units_sold"].to_numpy(dtype=np.float64))
    week_indices = _calendar_week_numbers(modeling_data["week_start"])

    try:
        dml_estimate, dml_se = _crossfit_dml(
            log_price,
            log_units,
            controls,
            week_indices,
            folds=folds,
            trees=trees,
            seed=2026,
        )
        ols_estimate, ols_se = _ols_log_log(
            log_price, log_units, controls, week_indices
        )
        naive_estimate, naive_se = _ols_log_log(
            log_price, log_units, np.empty((len(log_price), 0)), week_indices
        )
    except ValueError as error:
        base["fit_status"] = f"not_estimable: {error}"
        return base

    dml_lower, dml_upper = _interval(dml_estimate, dml_se)
    ols_lower, ols_upper = _interval(ols_estimate, ols_se)
    naive_lower, naive_upper = _interval(naive_estimate, naive_se)
    demand_type, recommendation = _recommendation(
        dml_estimate, dml_lower, dml_upper
    )
    base.update(
        {
            "elasticity": dml_estimate,
            "conf_interval": f"[{dml_lower:.3f}, {dml_upper:.3f}]",
            "ci_lower": dml_lower,
            "ci_upper": dml_upper,
            "naive_elasticity": naive_estimate,
            "naive_conf_interval": f"[{naive_lower:.3f}, {naive_upper:.3f}]",
            "ols_elasticity": ols_estimate,
            "ols_conf_interval": f"[{ols_lower:.3f}, {ols_upper:.3f}]",
            "demand_type": demand_type,
            "pricing_recommendation": recommendation,
            "fit_status": "estimated",
        }
    )
    if dml_estimate > 0:
        if dml_lower > 0:
            explanation = (
                "The adjusted estimate is significantly positive, which is not "
                "a normal demand response; investigate confounding, stock "
                "availability, sparse price changes, and the inferred promotion "
                "proxy before acting."
            )
        else:
            explanation = (
                "The point estimate is positive but its interval includes zero; "
                "the data do not establish a positive price response."
            )
        base["positive_elasticity_explanation"] = explanation
    return base


def _comparison_report(
    results: pd.DataFrame,
    selection: pd.DataFrame,
    output_path: Path,
    *,
    folds: int,
    trees: int,
    min_weeks: int,
    min_positive_weeks: int,
    min_price_levels: int,
    min_price_range: float,
) -> None:
    estimated = results.loc[results["fit_status"] == "estimated"]
    lines = [
        "# Module 2: price elasticity results",
        "",
        "## Method",
        "",
        "The dataset is aggregated to Saturday–Friday retail weeks to match M5's "
        "weekly price grain. The primary estimate is a cross-fitted "
        f"({folds}-fold), "
        "partialling-out Double Machine Learning (DML) estimate: Extra Trees "
        f"models ({trees} trees per fold model) predict log price and "
        "log1p(units_sold) "
        "from annual seasonality, SNAP exposure, event weeks, and the inferred "
        "promotion-week flag; the elasticity is fitted "
        "on the out-of-fold residuals. A naive log-log OLS model uses the same "
        "weekly observations without controls, while controlled OLS uses those "
        "calendar and promotion controls. All intervals use a "
        "four-week Newey-West adjustment.",
        "",
        "Zero-sales weeks are retained using log1p(units_sold). The "
        "selection step requires at least "
        f"{min_weeks} observed weeks, {min_positive_weeks} positive-sales "
        f"weeks, {min_price_levels} distinct prices, and a price range of at "
        f"least {min_price_range:.0%}. Up to {len(selection)} highest-volume "
        "eligible item-store combinations were selected.",
        "",
        "## DML vs. naive and controlled OLS",
        "",
    ]
    if estimated.empty:
        lines += [
            "No selected combination produced an estimable model. Review the "
            "fit_status column in the results table and relax eligibility "
            "criteria only if the data support doing so.",
            "",
        ]
    else:
        difference = estimated["elasticity"] - estimated["ols_elasticity"]
        naive_difference = estimated["elasticity"] - estimated["naive_elasticity"]
        correlation = estimated[["elasticity", "ols_elasticity"]].corr().iloc[0, 1]
        naive_correlation = estimated[["elasticity", "naive_elasticity"]].corr().iloc[0, 1]
        corr_text = f"{correlation:.2f}" if np.isfinite(correlation) else "not available"
        naive_corr_text = (
            f"{naive_correlation:.2f}" if np.isfinite(naive_correlation) else "not available"
        )
        lines += [
            f"Models were estimable for {len(estimated)} of {len(selection)} "
            f"selected combinations. Median DML elasticity was "
            f"{estimated['elasticity'].median():.3f}; median naive (uncontrolled) "
            f"log-log OLS elasticity was {estimated['naive_elasticity'].median():.3f}; "
            f"median controlled OLS elasticity was "
            f"{estimated['ols_elasticity'].median():.3f}.",
            "",
            f"**DML vs. naive:** the median absolute difference was "
            f"{naive_difference.abs().median():.3f} and the correlation was "
            f"{naive_corr_text}. The naive model uses no controls, so it mixes "
            "price response with seasonality, SNAP days, events and promotion "
            "timing.",
            "",
            f"**DML vs. controlled OLS:** the median absolute difference was "
            f"{difference.abs().median():.3f} and the correlation was "
            f"{corr_text}. Both adjust for the same observed controls; DML "
            "fits flexible nuisance models for price and demand, while "
            "controlled OLS imposes linear control effects. The inferred "
            "promotion flag is derived from price, so controlling for it also "
            "removes deep-discount price variation from the treatment.",
            "",
            "These differences show sensitivity to model specification, not "
            "proof that DML has removed all bias; estimates may still be "
            "unstable when price changes are sparse or predictable from time.",
            "",
            "The median 95% DML confidence-interval width was "
            f"{(estimated['ci_upper'] - estimated['ci_lower']).median():.3f}.",
            "",
            "The DML classifications were "
            f"{int((estimated['demand_type'] == 'Price-sensitive').sum())} "
            "price-sensitive, "
            f"{int((estimated['demand_type'] == 'Less sensitive').sum())} "
            "less-sensitive, and "
            f"{int((estimated['demand_type'] == 'Unclear').sum())} unclear. "
            "Unclear estimates are deliberately assigned no action.",
            "",
            "These estimates are observational associations and require "
            "no-important-unmeasured-confounding and adequate independent "
            "price variation for a causal interpretation. Price changes can "
            "coincide with unobserved demand or stock changes, and M5's "
            "promotion flag is inferred rather than source-labelled. Treat "
            "recommendations as hypotheses for controlled tests, not automatic "
            "price changes.",
            "",
        ]
    lines += [
        "## Output files",
        "",
        f"- Results: `{output_path.name}`",
        f"- Selected combinations: `{output_path.stem}_selected_combinations.csv`",
        f"- Audit: `{output_path.stem}_audit.json`",
        "",
        "A positive or statistically unclear DML elasticity is classified as "
        "unclear and receives no pricing action. Positive point estimates "
        "include an explanatory note in the results table.",
        "",
    ]
    output_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def estimate_price_elasticity(
    input_path: Path,
    output_path: Path,
    *,
    top_n: int = 100,
    min_weeks: int = 104,
    min_positive_weeks: int = 52,
    min_price_levels: int = 3,
    min_price_range: float = 0.05,
    folds: int = 5,
    trees: int = 50,
) -> dict[str, object]:
    """Select eligible item-store series, estimate elasticities, and save audit outputs."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Module 1 Parquet input not found: {input_path}")
    if not 1 <= top_n <= 200:
        raise ValueError("top_n must be between 1 and 200")
    if min_weeks < 1 or min_positive_weeks < 1 or min_price_levels < 2:
        raise ValueError("Week thresholds must be positive and price levels at least 2")
    if not 0 <= min_price_range < 1:
        raise ValueError("min_price_range must be in [0, 1)")
    if folds < 2 or trees < 1:
        raise ValueError("folds must be at least 2 and trees at least 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        columns = {
            row[0]
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(input_path)]
            ).fetchall()
        }
        required = {
            "item_id",
            "store_id",
            "date",
            "units_sold",
            "price",
            "snap_flag",
            "event_name",
            "promo_flag",
        }
        missing = sorted(required.difference(columns))
        if missing:
            raise ValueError(
                "Module 1 Parquet is missing required columns: " + ", ".join(missing)
            )

        selection = connection.execute(
            SUMMARY_SQL,
            [
                str(input_path),
                min_weeks,
                min_positive_weeks,
                min_price_levels,
                min_price_range,
                top_n,
            ],
        ).df()
        if selection.empty:
            raise ValueError(
                "No item-store combinations meet the price-variation and "
                "minimum-sales criteria. Review the criteria in assumptions.md."
            )

        connection.register("shortlisted", selection[["item_id", "store_id"]])
        weekly = connection.execute(WEEKLY_SQL, [str(input_path)]).df()
    finally:
        connection.close()

    results: list[dict[str, object]] = []
    selection_index = selection.set_index(["item_id", "store_id"], drop=False)
    for (item_id, store_id), group in weekly.groupby(
        ["item_id", "store_id"], sort=False, observed=True
    ):
        results.append(
            _estimate_combination(
                group,
                selection_index.loc[(item_id, store_id)],
                folds=folds,
                trees=trees,
            )
        )

    result_frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    result_frame.to_csv(output_path, index=False)
    selection_path = output_path.with_name(
        f"{output_path.stem}_selected_combinations.csv"
    )
    selection.to_csv(selection_path, index=False)
    audit: dict[str, object] = {
        "source_file": str(input_path),
        "results_file": str(output_path),
        "selection_file": str(selection_path),
        "selected_combinations": len(selection),
        "estimated_combinations": int((result_frame["fit_status"] == "estimated").sum()),
        "not_estimable_combinations": int(
            (result_frame["fit_status"] != "estimated").sum()
        ),
        "selection_criteria": {
            "top_n": top_n,
            "minimum_observed_weeks": min_weeks,
            "minimum_positive_sales_weeks": min_positive_weeks,
            "minimum_distinct_prices": min_price_levels,
            "minimum_price_range_pct": min_price_range,
        },
        "model": {
            "method": "Cross-fitted partialling-out DML",
            "nuisance_learner": "ExtraTreesRegressor",
            "folds": folds,
            "trees_per_fold_model": trees,
            "outcome_transform": "log1p(units_sold), including zero-sales weeks",
            "controls": [
                "annual sine seasonality",
                "annual cosine seasonality",
                "SNAP share",
                "event week",
                "inferred promotion week",
            ],
            "time_trend_control": False,
            "ols_baselines": [
                "Uncontrolled log-log OLS",
                "Controlled log-log OLS using DML controls",
            ],
            "confidence_interval": "95% Newey-West HAC, maximum lag 4 weeks",
        },
        "medians": {
            "dml_elasticity": float(result_frame["elasticity"].median()),
            "uncontrolled_ols_elasticity": float(
                result_frame["naive_elasticity"].median()
            ),
            "controlled_ols_elasticity": float(
                result_frame["ols_elasticity"].median()
            ),
            "dml_confidence_interval_width": float(
                (result_frame["ci_upper"] - result_frame["ci_lower"]).median()
            ),
        },
    }
    audit_path = output_path.with_name(f"{output_path.stem}_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _comparison_report(
        result_frame,
        selection,
        output_path,
        folds=folds,
        trees=trees,
        min_weeks=min_weeks,
        min_positive_weeks=min_positive_weeks,
        min_price_levels=min_price_levels,
        min_price_range=min_price_range,
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate M5 item-store price elasticities."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/module2/price_elasticities.csv"),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-weeks", type=int, default=104)
    parser.add_argument("--min-positive-weeks", type=int, default=52)
    parser.add_argument("--min-price-levels", type=int, default=3)
    parser.add_argument("--min-price-range", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=50)
    args = parser.parse_args()
    audit = estimate_price_elasticity(
        args.input,
        args.output,
        top_n=args.top_n,
        min_weeks=args.min_weeks,
        min_positive_weeks=args.min_positive_weeks,
        min_price_levels=args.min_price_levels,
        min_price_range=args.min_price_range,
        folds=args.folds,
        trees=args.trees,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
