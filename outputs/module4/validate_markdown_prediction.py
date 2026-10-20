"""Test whether markdown fields add held-out predictive signal in synthetic D2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "markdown" / "SYNTHETIC Markdown Dataset.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "module4" / "markdown_prediction_validation.md"
PLOT_PATH = PROJECT_ROOT / "outputs" / "module4" / "markdown_prediction_validation.png"
RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "module4"
    / "markdown_prediction_validation.csv"
)

CATEGORICAL_FEATURES = ["Category", "Brand", "Season"]
NUMERIC_FEATURES = [
    "Original_Price",
    "Competitor_Price",
    "Seasonality_Factor",
    "Historical_Sales",
]
TARGET_BASE_COLUMNS = {*CATEGORICAL_FEATURES, *NUMERIC_FEATURES}
STAGES = [
    (f"Markdown_{number}", f"Sales_After_M{number}")
    for number in range(1, 5)
]


def _model(feature_columns: list[str]) -> object:
    categorical = [name for name in feature_columns if name in CATEGORICAL_FEATURES]
    numeric = [name for name in feature_columns if name in NUMERIC_FEATURES or name.startswith("Markdown_")]
    transformations = []
    if categorical:
        transformations.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    if numeric:
        transformations.append(("numeric", "passthrough", numeric))
    preprocessor = ColumnTransformer(transformations)
    return make_pipeline(
        preprocessor,
        ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=15,
            max_features=0.9,
            random_state=20261018,
            n_jobs=-1,
        ),
    )


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = TARGET_BASE_COLUMNS | {
        "Product_ID",
        "Historical_Sales",
        *(column for pair in STAGES for column in pair),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    analysis = frame.drop_duplicates().reset_index(drop=True)
    if analysis["Product_ID"].duplicated().any():
        raise ValueError("Conflicting Product_ID rows remain after exact deduplication")
    if analysis["Historical_Sales"].le(0).any():
        raise ValueError("Historical_Sales must be positive")

    train_indices, test_indices = train_test_split(
        np.arange(len(analysis)),
        test_size=0.20,
        random_state=20261018,
    )
    rows: list[dict[str, float | int | str]] = []
    for stage_number, (markdown_column, sales_column) in enumerate(STAGES, start=1):
        target = (
            analysis[sales_column] - analysis["Historical_Sales"]
        ) / analysis["Historical_Sales"]
        baseline_prediction = np.full(
            len(test_indices),
            float(target.iloc[train_indices].median()),
        )
        feature_sets = {
            "metadata_only": sorted(TARGET_BASE_COLUMNS),
            "metadata_plus_markdown": sorted(TARGET_BASE_COLUMNS | {markdown_column}),
        }
        for model_name, feature_columns in feature_sets.items():
            estimator = _model(feature_columns)
            estimator.fit(
                analysis.iloc[train_indices][feature_columns],
                target.iloc[train_indices],
            )
            prediction = estimator.predict(analysis.iloc[test_indices][feature_columns])
            actual = target.iloc[test_indices].to_numpy()
            rows.append(
                {
                    "stage": f"M{stage_number}",
                    "model": model_name,
                    "train_rows": int(len(train_indices)),
                    "test_rows": int(len(test_indices)),
                    "baseline_median_test_mae": float(
                        mean_absolute_error(actual, baseline_prediction)
                    ),
                    "test_mae": float(mean_absolute_error(actual, prediction)),
                    "test_rmse": float(
                        np.sqrt(mean_squared_error(actual, prediction))
                    ),
                    "test_r2": float(r2_score(actual, prediction)),
                }
            )
    results = pd.DataFrame(rows)
    baseline = results.loc[results["model"].eq("metadata_only")].set_index("stage")
    with_markdown = results.loc[
        results["model"].eq("metadata_plus_markdown")
    ].set_index("stage")
    results["mae_improvement_vs_metadata"] = results.apply(
        lambda row: float(
            baseline.loc[row["stage"], "test_mae"] - row["test_mae"]
        )
        if row["model"] == "metadata_plus_markdown"
        else 0.0,
        axis=1,
    )
    results["r2_improvement_vs_metadata"] = results.apply(
        lambda row: float(
            row["test_r2"] - baseline.loc[row["stage"], "test_r2"]
        )
        if row["model"] == "metadata_plus_markdown"
        else 0.0,
        axis=1,
    )
    return results


def _render_report(results: pd.DataFrame, row_count: int, source_hash: str) -> str:
    table_rows = []
    for stage in ("M1", "M2", "M3", "M4"):
        stage_results = results.loc[results["stage"].eq(stage)].set_index("model")
        baseline = stage_results.loc["metadata_only"]
        augmented = stage_results.loc["metadata_plus_markdown"]
        table_rows.append(
            f"| {stage} | {baseline.test_mae * 100:.2f} | "
            f"{augmented.test_mae * 100:.2f} | "
            f"{augmented.mae_improvement_vs_metadata * 100:+.2f} | "
            f"{baseline.test_r2:.3f} | {augmented.test_r2:.3f} | "
            f"{augmented.r2_improvement_vs_metadata:+.3f} |"
        )

    better_stages = results.loc[
        results["model"].eq("metadata_plus_markdown")
        & results["mae_improvement_vs_metadata"].gt(0),
        "stage",
    ].tolist()
    if len(better_stages) == 4:
        summary = (
            "Adding each stage's matched markdown feature improved held-out MAE "
            "on this fixed split."
        )
    elif better_stages:
        summary = (
            "Adding the matched markdown feature improved held-out MAE for "
            f"{', '.join(better_stages)} on this fixed split, but not for every stage."
        )
    else:
        summary = (
            "Adding the matched markdown feature did not improve held-out MAE "
            "for any stage on this fixed split."
        )
    return f"""# Module 4 — held-out predictive check

## Method and data boundary

- Source: `data/raw/markdown/SYNTHETIC Markdown Dataset.csv`
- SHA-256: `{source_hash}`
- Exact duplicate rows were removed; analysis rows: {row_count:,}.
- Split: one reproducible 80/20 random product-level train/test split
  (`random_state=20261018`).
- Target: relative difference `(Sales_After_Mi - Historical_Sales) /
  Historical_Sales` for each stage.
- Learner: Extra Trees regression (`n_estimators=160`,
  `min_samples_leaf=15`, `max_features=0.9`).
- Compared models: metadata-only and metadata plus that stage's matched
  markdown field. The metadata is category, brand, season, original price,
  competitor price, seasonality factor, and historical sales.
- Deliberately excluded: `Product_ID`, all other stage markdown/sales fields,
  `Stock_Level`, `Customer Ratings`, `Return Rate`, `Promotion_Type`, and
  `Optimal Discount`. Fields with unclear measurement timing are not used as
  predictors; excluding `Optimal Discount` avoids target leakage.

## Test-set results

MAE is shown in percentage points of relative sales difference. Negative
improvement means the markdown-augmented model had a higher (worse) MAE.

| Stage | Metadata-only MAE (pp) | Plus markdown MAE (pp) | MAE improvement (pp) | Metadata-only R² | Plus markdown R² | R² improvement |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

{summary}

This is one held-out split of a synthetic, cross-sectional dataset. Predictive
improvement does not estimate the causal impact of markdowns, validate the
supplied sales measurement windows, establish profitability, or justify a
discount recommendation. Use the result only as a modelling diagnostic; a
real decision model requires dated outcomes, documented stage windows,
aligned inventory, and unit cost/margin data.
"""


def _save_plot(results: pd.DataFrame) -> None:
    labels = ["M1", "M2", "M3", "M4"]
    metadata = results.loc[results["model"].eq("metadata_only")].set_index("stage")
    augmented = results.loc[
        results["model"].eq("metadata_plus_markdown")
    ].set_index("stage")
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.bar(
        x - width / 2,
        [metadata.loc[label, "test_mae"] * 100 for label in labels],
        width,
        color="#1769AA",
        label="Metadata only",
    )
    axis.bar(
        x + width / 2,
        [augmented.loc[label, "test_mae"] * 100 for label in labels],
        width,
        color="#D97732",
        label="Metadata + matched markdown",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Held-out MAE (percentage points; lower is better)")
    axis.set_title("Synthetic D2 predictive check — one fixed holdout split")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Markdown dataset not found: {INPUT_PATH}")
    raw = INPUT_PATH.read_bytes()
    frame = pd.read_csv(INPUT_PATH)
    analysis = frame.drop_duplicates()
    results = validate_predictions(frame)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    REPORT_PATH.write_text(
        _render_report(
            results,
            len(analysis),
            hashlib.sha256(raw).hexdigest(),
        ),
        encoding="utf-8",
    )
    _save_plot(results)
    print(results.to_string(index=False))
    print(f"Report: {REPORT_PATH}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
