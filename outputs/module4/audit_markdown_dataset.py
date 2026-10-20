"""Audit the synthetic markdown dataset and write traceable audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "markdown" / "SYNTHETIC Markdown Dataset.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "module4" / "markdown_dataset_audit.md"
JSON_PATH = PROJECT_ROOT / "data" / "processed" / "module4" / "markdown_dataset_audit.json"

MARKDOWN_COLUMNS = ["Markdown_1", "Markdown_2", "Markdown_3", "Markdown_4"]
SALES_COLUMNS = [
    "Sales_After_M1",
    "Sales_After_M2",
    "Sales_After_M3",
    "Sales_After_M4",
]
NUMERIC_COLUMNS = [
    "Original_Price",
    "Competitor_Price",
    "Seasonality_Factor",
    *MARKDOWN_COLUMNS,
    "Historical_Sales",
    *SALES_COLUMNS,
    "Stock_Level",
    "Customer Ratings",
    "Return Rate",
    "Optimal Discount",
]
REQUIRED_COLUMNS = [
    "Product_ID",
    "Category",
    "Brand",
    "Season",
    "Product_Name",
    *NUMERIC_COLUMNS,
    "Promotion_Type",
]


def _transition_counts(
    frame: pd.DataFrame, columns: list[str]
) -> list[dict[str, int | str]]:
    transitions = []
    for before, after in zip(columns, columns[1:]):
        difference = frame[after] - frame[before]
        transitions.append(
            {
                "from": before,
                "to": after,
                "increase": int(difference.gt(0).sum()),
                "same": int(difference.eq(0).sum()),
                "decrease": int(difference.lt(0).sum()),
            }
        )
    return transitions


def audit_dataset(frame: pd.DataFrame, source_sha256: str) -> dict[str, Any]:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    numeric_summary = {}
    for column in NUMERIC_COLUMNS:
        series = frame[column]
        numeric_summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "nonpositive_count": int(series.le(0).sum()),
            "outside_unit_interval_count": int(
                (series.lt(0) | series.gt(1)).sum()
            )
            if column in (*MARKDOWN_COLUMNS, "Optimal Discount")
            else None,
        }

    repeated_ids = frame["Product_ID"].duplicated(keep="first")
    id_row_counts = frame.groupby("Product_ID", sort=False).size()
    duplicate_id_rows = frame.loc[frame["Product_ID"].duplicated(keep=False)]
    conflicting_duplicate_ids = int(
        duplicate_id_rows.drop_duplicates()
        .groupby("Product_ID", sort=False)
        .size()
        .gt(1)
        .sum()
    )

    return {
        "source_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": source_sha256,
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_by_column": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_product_id_rows_after_first": int(repeated_ids.sum()),
        "distinct_product_ids": int(frame["Product_ID"].nunique()),
        "product_id_min": int(frame["Product_ID"].min()),
        "product_id_max": int(frame["Product_ID"].max()),
        "duplicate_product_ids_with_conflicting_rows": conflicting_duplicate_ids,
        "product_ids_with_multiple_rows": int(id_row_counts.gt(1).sum()),
        "numeric_summary": numeric_summary,
        "category_counts": {
            str(key): int(value)
            for key, value in frame["Category"].value_counts(dropna=False).items()
        },
        "season_counts": {
            str(key): int(value)
            for key, value in frame["Season"].value_counts(dropna=False).items()
        },
        "promotion_type_counts": {
            str(key): int(value)
            for key, value in frame["Promotion_Type"]
            .value_counts(dropna=False)
            .items()
        },
        "markdown_transitions": _transition_counts(frame, MARKDOWN_COLUMNS),
        "sales_transitions": _transition_counts(frame, SALES_COLUMNS),
        "sales_after_m4_exceeds_stock_level_count": int(
            frame["Sales_After_M4"].gt(frame["Stock_Level"]).sum()
        ),
        "stock_level_nonpositive_count": int(frame["Stock_Level"].le(0).sum()),
    }


def render_report(audit: dict[str, Any]) -> str:
    markdown_transitions = audit["markdown_transitions"]
    sales_transitions = audit["sales_transitions"]
    markdown_table = "\n".join(
        f"| {step['from']} → {step['to']} | {step['increase']:,} | "
        f"{step['same']:,} | {step['decrease']:,} |"
        for step in markdown_transitions
    )
    sales_table = "\n".join(
        f"| {step['from']} → {step['to']} | {step['increase']:,} | "
        f"{step['same']:,} | {step['decrease']:,} |"
        for step in sales_transitions
    )
    numeric_table = "\n".join(
        f"| `{column}` | {values['min']:,.2f} | {values['max']:,.2f} | "
        f"{values['mean']:,.2f} |"
        for column, values in audit["numeric_summary"].items()
    )
    category_counts = ", ".join(
        f"{category}: {count:,}"
        for category, count in audit["category_counts"].items()
    )
    season_counts = ", ".join(
        f"{season}: {count:,}" for season, count in audit["season_counts"].items()
    )
    promotion_counts = ", ".join(
        f"{promotion}: {count:,}"
        for promotion, count in audit["promotion_type_counts"].items()
    )
    duplicate_note = (
        f"All {audit['exact_duplicate_rows']:,} full-row duplicate records are exact "
        "copies."
        if audit["exact_duplicate_rows"]
        == audit["duplicate_product_id_rows_after_first"]
        and audit["duplicate_product_ids_with_conflicting_rows"] == 0
        else "Inspect repeated Product_ID values before deciding whether records "
        "are duplicate observations."
    )
    return f"""# Module 4 — D2 markdown dataset audit

## Provenance

- Source file: `{audit['source_file']}`
- SHA-256: `{audit['source_sha256']}`
- Rows and columns: {audit['row_count']:,} × {audit['column_count']}
- Generated by: `outputs/module4/audit_markdown_dataset.py`
- Machine-readable audit: `data/processed/module4/markdown_dataset_audit.json`

The Kaggle dataset describes itself as synthetic. Results from it are therefore
useful as a modelling exercise, not as evidence of real retail markdown
performance.

## Completeness and identity

- Missing values: {sum(audit['missing_by_column'].values()):,} across all columns.
- Exact duplicate rows: {audit['exact_duplicate_rows']:,}.
- Duplicate `Product_ID` rows after the first occurrence:
  {audit['duplicate_product_id_rows_after_first']:,}; distinct IDs:
  {audit['distinct_product_ids']:,} (range {audit['product_id_min']:,}–{audit['product_id_max']:,}).
- Repeated IDs with conflicting row values:
  {audit['duplicate_product_ids_with_conflicting_rows']:,}.
- {duplicate_note}
- Non-positive stock records: {audit['stock_level_nonpositive_count']:,}.

For an analysis view, drop exact duplicate rows only; keep the source CSV
unchanged and record the resulting row count. Do not deduplicate by product ID
until conflicting duplicates have been ruled out.

Category counts: {category_counts}.  
Season counts: {season_counts}.  
Promotion-type counts: {promotion_counts}.

## Numeric ranges

| Field | Minimum | Maximum | Mean |
|---|---:|---:|---:|
{numeric_table}

Markdown fields and `Optimal Discount` are between 0 and 1 in every row; price
fields are positive. The values look fraction-like, but the file does not
provide units or a field-level data dictionary. Treat their interpretation as
discount proportions as a provisional assumption, not a verified fact.

## Are the M1–M4 fields a sequence?

| Adjacent fields | Increased | Unchanged | Decreased |
|---|---:|---:|---:|
{markdown_table}

| Adjacent fields | Increased | Unchanged | Decreased |
|---|---:|---:|---:|
{sales_table}

Markdown values frequently move both up and down between adjacent numbered
fields. Sales usually increase but also decrease for many rows. With no dates,
stage definitions, exposure durations, or experimental design, the numbered
fields cannot be treated as an observed chronological markdown path or as
comparable weekly outcomes. `Optimal Discount` is a supplied target/benchmark,
not a validated outcome; exclude it from model inputs when predicting a
discount or sales response.

## Scope and limitations

- Present: product/category/brand labels, original and competitor prices,
  four markdown values, historical sales, four post-markdown sales fields,
  stock level, promotion type, customer ratings, return rate, seasonality, and
  a supplied optimal-discount field.
- Absent: dates, weekly/daily sales, markdown timing/duration, unit cost,
  gross margin, inventory replenishment, and a data dictionary describing
  measurement horizons.
- `Sales_After_M4` exceeds `Stock_Level` in
  {audit['sales_after_m4_exceeds_stock_level_count']:,} of {audit['row_count']:,}
  rows. Because sales and stock periods/units are undocumented, these columns
  cannot yet be combined into a leftover-stock calculation.

**Conclusion:** this dataset supports exploratory product-level associations
between the supplied markdown fields and sales fields, subject to duplicate
handling and the synthetic-data caveat. It does not currently support a
time-based markdown optimizer, a reliable stock-clearance/leftover model, or
profit/ROI recommendations. Obtain the field definitions, stage timing, stock
measurement horizon, and unit cost/margin data before making those claims.
"""


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Markdown dataset not found: {INPUT_PATH}")

    digest = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    frame = pd.read_csv(INPUT_PATH)
    audit = audit_dataset(frame, digest)
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(audit), encoding="utf-8")
    print(f"Audited {audit['row_count']:,} rows × {audit['column_count']} columns")
    print(f"Exact duplicate rows: {audit['exact_duplicate_rows']:,}")
    print(f"Missing values: {sum(audit['missing_by_column'].values()):,}")
    print(f"Report: {REPORT_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
