# Retail Price & Promotion Decision Engine

An end-to-end, reproducible retail analytics prototype for price, promotion,
markdown, and bundle decisions using M5, D2, and Instacart data. It produces
audited datasets, model diagnostics, and reviewable recommendations—not
automatic commercial actions.

## Limitations

- M5 has no promotion label; promotions are inferred from historical price drops.
- Promotion margin rates, markdown cost/salvage values, thresholds, and other
  business rules are documented assumptions, not observed facts.
- D2 identifies itself as synthetic and lacks documented sales and inventory
  measurement periods; markdown recommendations are scenarios, not validated
  retail outcomes.
- M5 elasticity estimates are observational and can be affected by unmeasured
  confounding; pooled defaults are department-level, not SKU-specific.
- Instacart contains no prices. Bundle candidates are associations and need
  cross-price validation before pricing or promotion decisions.

## Project layout

```text
data/
  raw/                 M5 CSVs here; D2 and Instacart in documented subfolders
  processed/           Generated Parquet and audit outputs (not committed)
outputs/               Review plots and query benchmark records
src/retail_engine/     Reusable analysis code and command-line entry point
tests/                 Focused automated tests
assumptions.md          Business and modelling assumptions
```

## Reproducing results

The raw and processed data are excluded from Git. Download the datasets from
their original sources and place the files as follows:

- **M5 (D1):** join the competition, accept its rules, and download
  [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
  Put `sales_train_evaluation.csv`, `sell_prices.csv`, and `calendar.csv` in
  `data/raw/`.
- **Markdown (D2):** download the
  [Retail Markdown Optimization dataset](https://www.kaggle.com/datasets/arbaaztamboli/retail-markdown-optimization-discounts-and-sales).
  Put the CSV at `data/raw/markdown/SYNTHETIC Markdown Dataset.csv`.
- **Instacart (D3):** download the
  [Instacart Market Basket Analysis dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis).
  Put `orders.csv`, `order_products__prior.csv`,
  `order_products__train.csv`, `products.csv`, `aisles.csv`, and
  `departments.csv` in `data/raw/instacart/`.

From an activated project virtual environment, install the project and run the
pipelines in dependency order:

```powershell
py -m pip install -e ".[dev]"
build-retail-data
estimate-price-elasticity
estimate-pooled-elasticity --max-series 10000 --by-store
estimate-promotion-uplift
optimize-markdowns --historical-sales-period-weeks 13 --horizon-weeks 8
estimate-bundles
```

Module 1 is the longest step: it reshapes tens of millions of daily rows.
Allow several GB of free memory and use `build-retail-data --chunk-skus 32` if
the default SKU batch size exceeds available memory. Module 4 results are
scenario outputs under documented assumptions; Module 5 results are
descriptive and do not replace cross-price validation. The optional C1
DoubleML validation workflow is documented separately under Module 2.

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
these are department-level defaults attached to item-store rows, not
SKU-specific elasticity estimates. A series is called price-sensitive only
when the upper 95% CI bound is below -1. The individual DML result is retained
only as a review flag and does not override the default action. Results, the
selected series and an audit JSON are written to
`data/processed/module2_pooled/`. Use `--max-series` to change the subset
size. Eligibility rules and limitations are recorded in
[assumptions.md](assumptions.md).

Module 2 interpretation artifacts, including the generated naive/DML/pooled
comparison, the authored
[`module2_interpretation.md`](outputs/module2/module2_interpretation.md),
category medians, and positive-elasticity diagnostic plots, are in
`outputs/module2/`. Recreate the five largest positive-series plots and their
diagnostic notes with:

```powershell
py -3.10 outputs/module2/generate_positive_elasticity_cases.py
```

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
compares sales with the mean of the four latest non-promotion weeks in the
preceding eight calendar weeks, and estimates incremental revenue and margin.
It uses assumed regular-price gross margins of 25% for FOODS, 35% for
HOUSEHOLD, and 40% for HOBBIES. These are assumptions, not data; pass
`--gross-margin-rate` only when intentionally overriding the category margins.
The output records each episode's category, average discount depth, and margin
rate used. Episodes without enough baseline weeks are retained and marked for
review, rather than silently omitted. The CSV and audit JSON are written to
`data/processed/module3/`. The calculation, recommendation rules, and
financial assumptions are documented in
`assumptions.md`; first-run totals and hand calculations are in
`outputs/module3/promotion_uplift_summary.md`.

### Module 4: markdown dataset audit

Place D2 at `data/raw/markdown/SYNTHETIC Markdown Dataset.csv`, then run the
data audit from the project root:

```powershell
py -3.10 outputs/module4/audit_markdown_dataset.py
```

The script records the source SHA-256, missingness, duplicates, ranges, and
stage-field movements. It writes the review report to
`outputs/module4/markdown_dataset_audit.md` and a machine-readable JSON to
`data/processed/module4/markdown_dataset_audit.json`. The source describes
itself as synthetic; fields have no dates or documented measurement horizons.
The simulator below uses user-specified scenario assumptions for cost, stock,
and the demand horizon; its output is not a validated business
recommendation. Module 4 assumptions and limits are in
[assumptions.md](assumptions.md).

The markdown simulator uses labelled scenario assumptions documented in
`assumptions.md`: its default interpretation is weekly historical and matched
stage sales, unit cost is 60% of original price, salvage is 20% of unit cost,
and the demand horizon is eight weeks. Because D2 does not document its sales
measurement period, the evidence run also tests 13- and 26-week period
assumptions against 4-, 8-, and 16-week horizons. Markdown choices use a
0/10/20/30/40/50% grid and are allowed only when stock cover strictly exceeds
the simulated horizon. These are scenario assumptions, not data facts. Run:

```powershell
optimize-markdowns --historical-sales-period-weeks 13 --horizon-weeks 8
```

> 13 weeks is an illustrative assumption; D2 does not document the sales
> period. See the period/horizon sensitivity table.

The command removes exact duplicate input rows, rejects conflicting duplicate
product IDs, and writes each product's maximum-predicted-profit markdown to
`data/processed/module4/markdown_recommendations.csv` along with an assumptions
and provenance audit JSON. Because the dataset is synthetic and its stage
durations and stock measurement horizon are undocumented, the outputs are
scenario results under these assumptions, not validated retail
recommendations. The audit includes fixed-30% comparisons only where the
selected markdown differs, and an `Optimal Discount` reference column that is
never used by the model. Evidence includes five SKU profit curves, a 3×3
cost/salvage sensitivity, and all nine period/horizon variants in
`outputs/module4/evidence/`; each is labelled “scenario under assumptions” to
reflect the synthetic source. Pass `--historical-sales-period-weeks 13` or `26`
to run the recommendation CSV under either alternate period assumption.

## Module status and roadmap

1. **Data foundation:** clean M5 item-store-day table (implemented).
2. **Price elasticity:** select price-varying SKU-store combinations, compare
   naive log-log regression with cross-fitted DML, and produce confidence
   intervals and cautious actions (M5 implementation, pooled diagnostics, and
   C1/DoubleML validation complete; interpretation and limits are documented).
3. **Promotion uplift:** estimated incremental units and profit against a
   documented baseline, with a two-week post-promotion dip check (implemented).
4. **Markdown optimization:** D2 source audited; a labelled-assumption
   product-level scenario simulator is implemented. Validated time-based,
   inventory, and profit optimization still require better field definitions
   and additional data.
5. **Market basket and bundles:** mine Instacart associations and mark
   cross-promotion validation as required rather than treating association as
   causal evidence.

Each module should retain its runnable source, derived outputs, and assumptions
so that every recommendation can be traced to data and code.

### Module 5: Instacart bundles and market basket analysis

Place the Instacart CSV files in `data/raw/instacart/`, install the project
dependencies, and run the bundle-candidate workflow:

```powershell
pip install -e .
estimate-bundles
```

The CLI runs from `src/retail_engine/module5.py` and writes
`data/processed/module5/bundle_candidates.csv`, the support/lift sensitivity
summary, a source audit, a report, and an mlxtend FP-Growth cross-check. It
defaults to at least 0.25% prior-order support and prior lift strictly above
1.5; override with `--min-support` or `--min-lift`. Three ranked pairs are
cross-checked against mlxtend FP-Growth on a deterministic 100,000-prior-order
sample. The department-to-M5 category crosswalk and validation caveats are
documented in [assumptions.md](assumptions.md).

“Bought together” is descriptive co-occurrence, not evidence that the products
respond to each other's price changes. Association does not establish
cross-price elasticity or causal bundle uplift; validate before acting.

### Local dashboard

The standalone dashboard is available at `outputs/dashboard/index.html`. Open
it directly in a browser; it has no build step or external runtime
dependencies. It presents a snapshot of the generated project artifacts, with
module-specific source paths and limitations. Its figures are embedded from
the documented run outputs and will not refresh automatically when those
outputs change.

Install the optional dashboard dependency and launch the five-tab dashboard
from the project root:

```powershell
py -3.10 -m pip install -e ".[dashboard]"
py -3.10 -m streamlit run outputs/dashboard/app.py --server.address 127.0.0.1
```

The app reads existing audit JSON, CSV, Parquet references, and interpretation
artifacts under `data/processed/` and `outputs/`; it does not regenerate model
outputs. Each module page shows its source files and limitations. Charts use
blue, purple, orange, pink, and red accents only.
