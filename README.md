# Retail Price & Promotion Decision Engine

A reproducible, data-backed project for retail pricing and promotion analysis.
The first milestone builds the shared item-store-day table from M5 and records
the assumptions, row counts, and promotion-proxy spot checks used to make it.

## Project layout

```text
data/
  raw/                 Put the three M5 CSV files here (not committed)
  processed/           Generated Parquet and audit outputs (not committed)
outputs/               Review plots and query benchmark records
src/retail_engine/     Reusable analysis code and command-line entry point
tests/                 Focused automated tests
assumptions.md          Business and modelling assumptions
```

## Module 1: data foundation

### 1. Get the M5 files

Join the M5 Forecasting competition on Kaggle, accept its rules, and download:

- `sales_train_evaluation.csv`
- `sell_prices.csv`
- `calendar.csv`

Place those files in `data/raw/`. The raw data is intentionally excluded from
version control.

### 2. Install and run

From the project root, in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[dev]"
build-retail-data
```

The default command writes:

- `data/processed/clean_retail_data.parquet`
- `data/processed/clean_retail_data_audit.json`
- `data/processed/clean_retail_data_promo_spot_checks.csv`
- `data/processed/clean_retail_data_promo_spot_checks.png`

The pipeline reshapes sales in bounded SKU-store batches rather than loading
the entire multi-year daily table into memory at once. Lower the batch size if
needed:

```powershell
build-retail-data --chunk-skus 32
```

Input and output paths can be overridden with `--sales`, `--prices`,
`--calendar`, and `--output`. The Parquet file is a single compressed file; its
row groups retain statistics that allow Parquet readers to skip irrelevant
data when filtering.

### 3. Validate

```powershell
py -m pytest
```

Review the audit JSON for dropped-row counts, the spot-check CSV for ten
SKU-store weekly price histories, and the plot for the prior-eight-week price
baseline and threshold-qualified drops. The exact proxy definition and its
limitations are in [assumptions.md](assumptions.md).

The ten-SKU plot and its source rows are also saved in `outputs/module1/` for
review. M5's sales source starts as 30,490 item-store records with 1,941 daily
sales columns; after reshaping and joining weekly prices and calendar fields,
the output has 46,881,677 active item-store-day rows. Of 59,181,090 possible
item-store-day rows, 12,299,413 (20.78%) had no matching weekly price and were
excluded. A single DuckDB query filtering `FOODS_1_001` at `CA_1` from
2015-01-01 through 2015-03-31 returned 90 rows and 58 units in 0.7209 seconds
on the current workspace; see `outputs/module1/query_benchmark.md`. This is
one observed run, not a performance guarantee.

## Module 2: price elasticity

After Module 1 has produced the Parquet table, install the updated project
dependencies and run:

```powershell
py -m pip install -e ".[dev]"
estimate-price-elasticity
```

This scans the Parquet data, chooses up to 100 highest-volume SKU-store series
with sufficient price variation, aggregates them to M5 retail weeks, and
compares cross-fitted partialling-out DML with controlled and uncontrolled
OLS. The outcome uses `log1p(units_sold)` so zero-sales weeks stay in the model;
controls include annual seasonality, SNAP, event weeks, and the inferred
promotion-week flag, with no linear time trend. Outputs are written under
`data/processed/module2/`: elasticity results, the selected
series list, an audit JSON, and a short comparison write-up. The thresholds,
controls, zero-sales handling, and interpretation caveats are listed in
[assumptions.md](assumptions.md). Recommendations are review/test hypotheses,
not automatic price changes.

**Learning checkpoint:** this first M5 implementation uses the partialling-out
DML structure but does not yet reproduce every step of the C1 notebook on its
separate Online Retail example dataset. That reference reproduction and its
learner/sensitivity comparisons remain to be completed before considering the
causal approach validated.

## Planned modules

1. **Data foundation:** clean M5 item-store-day table (implemented).
2. **Price elasticity:** select price-varying SKU-store combinations, compare
   naive log-log regression with cross-fitted DML, and produce confidence
   intervals and cautious actions (M5 implementation complete; C1 reference
   reproduction remains).
3. **Promotion uplift:** estimate incremental units and profit against a
   documented baseline, then assess post-promotion dips.
4. **Markdown optimization:** simulate markdown choices and leftover salvage
   value using the separate markdown dataset.
5. **Market basket and bundles:** mine Instacart associations and mark
   cross-promotion validation as required rather than treating association as
   causal evidence.

Each module should retain its runnable source, derived outputs, and assumptions
so that every recommendation can be traced to data and code.
