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

**Learning checkpoint:** the M5 implementation uses the partialling-out DML
structure. The separate C1 Online Retail reference exercise is also complete:
four nuisance learners were compared, and a selected M5 series was
cross-checked against `DoubleMLPLR`. These exercises validate the implementation
workflow, not the causal identification assumptions or the generalizability of
the M5 estimates.

### Pooled department fixed-effects model

For a pooled diagnostic that borrows price variation across products, run:

```powershell
estimate-pooled-elasticity --max-series 10000 --by-store
```

The default run selects up to 2,000 eligible item-store series, balanced as
evenly as possible across the seven departments, and estimates two models per
department. Both use item-store and store-week fixed effects and standard
errors clustered by item-store. The log-linear model uses
`log(units_sold)` on `log(price)` and excludes zero-sales weeks. PPML models
units sold using `log(price)` and retains zero-sales weeks. PPML is the primary
pooled comparison; the log-linear estimate is included as a sensitivity check.
The `--by-store` option also fits department models separately within each
store. Per-item-store pricing defaults use the pooled department PPML estimate;
the individual DML result is retained only as a review flag and does not
override the default action. Results, the selected series and an audit JSON
are written to `data/processed/module2_pooled/`. Use `--max-series` to change
the subset size. Eligibility rules and limitations are recorded in
[assumptions.md](assumptions.md).

### Module 2 validation

Install the optional DoubleML validation dependency and download the C1
preprocessed `elasticity_subset.csv` from
<https://raw.githubusercontent.com/DoubleML/doubleml-docs/main/doc/examples/data/elasticity_subset.csv>
to `data/raw/elasticity_subset.csv`, then run:

```powershell
py -m pip install -e ".[validation]"
validate-module2
```

This runs the C1 PLR example with linear, LassoCV, random-forest, and
histogram-gradient-boosting nuisance learners, evaluates their nuisance
prediction performance, and cross-checks a selected M5 series against
`DoubleMLPLR`. The input data and generated validation outputs remain under
ignored `data/` paths: `data/processed/module2_validation/` contains the
learner results CSV, audit JSON, and M5 cross-check JSON. The recorded results
and interpretation are summarized in
`outputs/module2/module2_elasticity_interpretation.md`.

### Module 3: promotion uplift

After Module 1 has produced the clean Parquet file, run:

```powershell
estimate-promotion-uplift
```

The command groups consecutive inferred promotion weeks for each item-store,
compares sales with the median of the four latest non-promotion weeks in the
preceding eight calendar weeks, and estimates incremental revenue and margin.
It assumes a 30% regular-price gross margin by default; pass
`--gross-margin-rate` to override it. Episodes without enough baseline weeks
are retained and marked for review, rather than silently omitted. The CSV and
audit JSON are written to `data/processed/module3/`. The calculation,
recommendation rules, and financial assumptions are documented in
`assumptions.md`; first-run totals and hand calculations are in
`outputs/module3/promotion_uplift_summary.md`.

## Module status and roadmap

1. **Data foundation:** clean M5 item-store-day table (implemented).
2. **Price elasticity:** select price-varying SKU-store combinations, compare
   naive log-log regression with cross-fitted DML, and produce confidence
   intervals and cautious actions (M5 implementation, pooled diagnostics, and
   C1/DoubleML validation complete; interpretation and limits are documented).
3. **Promotion uplift:** estimated incremental units and profit against a
   documented baseline, with a two-week post-promotion dip check (implemented).
4. **Markdown optimization:** simulate markdown choices and leftover salvage
   value using the separate markdown dataset.
5. **Market basket and bundles:** mine Instacart associations and mark
   cross-promotion validation as required rather than treating association as
   causal evidence.

Each module should retain its runnable source, derived outputs, and assumptions
so that every recommendation can be traced to data and code.
