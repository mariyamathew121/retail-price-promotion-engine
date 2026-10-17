"""Reproduce the DoubleML elasticity example and cross-check Module 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression

from retail_engine.module2 import (
    WEEK_START_SQL,
    _crossfit_dml,
    _weekly_control_matrix,
)

C1_DATA_URL = (
    "https://raw.githubusercontent.com/DoubleML/doubleml-docs/main/"
    "doc/examples/data/elasticity_subset.csv"
)
C1_EXCLUDED_COLUMNS = {"Quantity", "revenue", "UnitPrice", "dLnQ", "dLnP"}


def _doubleml_classes():
    try:
        from doubleml import DoubleMLData, DoubleMLPLR
    except ImportError as error:
        raise ImportError(
            "DoubleML validation requires the optional dependency; install "
            'the project with `python -m pip install -e ".[validation]"`.'
        ) from error
    return DoubleMLData, DoubleMLPLR


def _new_c1_learners(seed: int) -> dict[str, tuple[object, object]]:
    return {
        "linear": (LinearRegression(), LinearRegression()),
        "lasso_cv": (
            LassoCV(cv=5, max_iter=10_000, n_jobs=1, random_state=seed),
            LassoCV(cv=5, max_iter=10_000, n_jobs=1, random_state=seed),
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                max_features=1.0,
                n_jobs=1,
                random_state=seed,
            ),
            RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                max_features=1.0,
                n_jobs=1,
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    }


def run_c1_replication(
    data_path: Path,
    output_dir: Path,
    *,
    seed: int = 123,
    n_folds: int = 5,
) -> pd.DataFrame:
    DoubleMLData, DoubleMLPLR = _doubleml_classes()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Tutorial subset not found at {data_path}. Download {C1_DATA_URL} "
            "to that path first."
        )
    data = pd.read_csv(data_path, index_col=0)
    required = C1_EXCLUDED_COLUMNS
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Tutorial data is missing required columns: {sorted(missing)}")

    feature_names = [column for column in data.columns if column not in required]
    x_data = data[feature_names].copy()
    for column in x_data.select_dtypes(include="bool").columns:
        x_data[column] = x_data[column].astype(np.int8)
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x_data.dtypes):
        raise ValueError("C1 nuisance features must be numeric after preprocessing")
    data_dml_frame = pd.concat(
        [
            data[["dLnQ", "dLnP"]].reset_index(drop=True),
            x_data.reset_index(drop=True),
        ],
        axis=1,
    )
    data_dml = DoubleMLData(
        data_dml_frame,
        y_col="dLnQ",
        d_cols="dLnP",
        x_cols=feature_names,
    )

    rows: list[dict[str, object]] = []
    for learner_name, (ml_l, ml_m) in _new_c1_learners(seed).items():
        np.random.seed(seed)
        model = DoubleMLPLR(
            data_dml,
            ml_l=ml_l,
            ml_m=ml_m,
            n_folds=n_folds,
            score="partialling out",
        )
        model.fit()
        summary = model.summary.loc["dLnP"]
        learner_errors = model.evaluate_learners()
        rows.append(
            {
                "learner": learner_name,
                "elasticity": float(summary["coef"]),
                "standard_error": float(summary["std err"]),
                "ci_lower": float(summary["2.5 %"]),
                "ci_upper": float(summary["97.5 %"]),
                "p_value": float(summary["P>|t|"]),
                "ml_l_rmse": float(np.mean(learner_errors["ml_l"])),
                "ml_m_rmse": float(np.mean(learner_errors["ml_m"])),
                "observations": int(data.shape[0]),
                "features": int(len(feature_names)),
            }
        )

    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "c1_doubleml_replication.csv", index=False)
    audit = {
        "source_url": C1_DATA_URL,
        "input_file": str(data_path),
        "rows": int(data.shape[0]),
        "features": int(len(feature_names)),
        "outcome": "dLnQ",
        "treatment": "dLnP",
        "n_folds": n_folds,
        "score": "partialling out",
        "seed": seed,
        "learners": list(_new_c1_learners(seed)),
        "results": results.to_dict(orient="records"),
    }
    (output_dir / "c1_doubleml_replication_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def run_m5_doubleml_crosscheck(
    input_path: Path,
    module2_results_path: Path,
    output_dir: Path,
    *,
    seed: int = 123,
    trees: int = 50,
    folds: int = 5,
) -> dict[str, object]:
    DoubleMLData, DoubleMLPLR = _doubleml_classes()
    results = pd.read_csv(module2_results_path)
    if results.empty:
        raise ValueError("Module 2 result file has no item-store combinations")
    selected = results.sort_values(
        ["item_id", "store_id"], kind="stable"
    ).iloc[0]
    con = duckdb.connect()
    weekly = con.execute(
        f"""
        SELECT {WEEK_START_SQL} AS week_start,
               SUM(units_sold)::DOUBLE AS units_sold,
               AVG(price)::DOUBLE AS price,
               AVG(snap_flag)::DOUBLE AS snap_share,
               MAX(CASE WHEN event_name IS NOT NULL AND event_name <> ''
                        THEN 1 ELSE 0 END)::DOUBLE AS event_week,
               MAX(promo_flag)::DOUBLE AS promo_week
        FROM read_parquet(?)
        WHERE item_id = ? AND store_id = ?
        GROUP BY week_start
        ORDER BY week_start
        """,
        [str(input_path), selected["item_id"], selected["store_id"]],
    ).df()
    con.close()
    if len(weekly) < folds * 5:
        raise ValueError("Selected M5 series has too few weeks for cross-fitting")

    controls = _weekly_control_matrix(weekly)
    log_price = np.log(weekly["price"].to_numpy(dtype=np.float64))
    log_units = np.log1p(weekly["units_sold"].to_numpy(dtype=np.float64))
    custom_estimate, custom_se = _crossfit_dml(
        log_price,
        log_units,
        controls,
        np.arange(len(weekly)),
        folds=folds,
        trees=trees,
        seed=seed,
    )
    np.random.seed(seed)
    data_dml = DoubleMLData.from_arrays(
        x=controls,
        y=log_units,
        d=log_price,
    )
    doubleml_model = DoubleMLPLR(
        data_dml,
        ml_l=ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        ),
        ml_m=ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=seed + 1,
            n_jobs=1,
        ),
        n_folds=folds,
        score="partialling out",
    )
    doubleml_model.fit()
    doubleml_summary = doubleml_model.summary.loc["d"]
    output = {
        "item_id": str(selected["item_id"]),
        "store_id": str(selected["store_id"]),
        "weeks": int(len(weekly)),
        "custom_consecutive_fold_estimate": custom_estimate,
        "custom_hac_standard_error": custom_se,
        "doubleml_random_fold_estimate": float(doubleml_summary["coef"]),
        "doubleml_standard_error": float(doubleml_summary["std err"]),
        "doubleml_ci_lower": float(doubleml_summary["2.5 %"]),
        "doubleml_ci_upper": float(doubleml_summary["97.5 %"]),
        "estimate_difference": (
            custom_estimate - float(doubleml_summary["coef"])
        ),
        "note": (
            "Point-estimate cross-check only: the custom estimator uses "
            "consecutive folds and HAC uncertainty; DoubleML uses random folds "
            "and its default iid inference."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m5_doubleml_crosscheck.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Module 2 with the DoubleML tutorial and a PLR cross-check."
    )
    parser.add_argument(
        "--c1-data",
        type=Path,
        default=Path("data/raw/elasticity_subset.csv"),
    )
    parser.add_argument(
        "--m5-input",
        type=Path,
        default=Path("data/processed/clean_retail_data.parquet"),
    )
    parser.add_argument(
        "--module2-results",
        type=Path,
        default=Path("data/processed/module2/price_elasticities.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/module2_validation"),
    )
    args = parser.parse_args()

    c1_results = run_c1_replication(args.c1_data, args.output_dir)
    crosscheck = run_m5_doubleml_crosscheck(
        args.m5_input,
        args.module2_results,
        args.output_dir,
    )
    print("C1 DoubleML tutorial replication:")
    print(c1_results.to_string(index=False))
    print("\nM5 custom-vs-DoubleMLPLR cross-check:")
    print(json.dumps(crosscheck, indent=2))


if __name__ == "__main__":
    main()
