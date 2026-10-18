# Module 3 promotion uplift — first M5 run

## Method and outputs

The analysis reads `data/processed/clean_retail_data.parquet` and is produced
by `src/retail_engine/module3.py`. It treats consecutive flagged Saturday-start
weeks for an item-store as one inferred promotion episode. The baseline is the
median unit sales from the four latest non-promotion weeks within the preceding
eight calendar weeks. The assumed regular-price gross margin is 30%; estimated
unit cost is held fixed through the promotion. See `assumptions.md` for the
full definitions and recommendation rules.

The generated, row-level table and its audit are
`data/processed/module3/promotion_uplift.csv` and
`data/processed/module3/promotion_uplift_audit.json`. Those reproducible data
artifacts are ignored by Git; this script and report document how they were
created.

## Run totals

| Result | Count |
|---|---:|
| Inferred promotion episodes | 16,749 |
| Fully evaluated (four baseline weeks available) | 16,253 |
| Marked for review (insufficient baseline) | 496 |
| Continue promotion | 4,594 |
| Reduce discount | 5,209 |
| Stop promotion | 6,450 |

Among complete episodes, median duration was five weeks; the 75th percentile
was eight weeks. These are runs in the Module 1 price-drop proxy, not verified
retailer campaigns. About 25.9% of complete episodes had a zero-unit baseline,
so sparse-history and inventory context should be checked before acting.
The two post-promotion weeks were available for 98.8% of complete episodes.

## Hand-calculated checks

`tests/test_module3.py::test_three_hand_calculated_promotion_examples` checks
the following arithmetic at a $10 regular price and assumed 30% gross margin
(unit cost $7). Each case has four baseline weeks at 10 units/week, two
promotion weeks, and two post-promotion weeks at 8 units/week.

| Case | Promotion price; units/week | Baseline sales | Promo sales | Incremental units | Incremental revenue | Incremental margin | Discount cost | ROI | Action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | $9; 20 | 20 | 40 | 20 | $160 | $20 | $40 | 0.50 | Continue |
| B | $8; 12 | 20 | 24 | 4 | -$8 | -$36 | $48 | -0.75 | Reduce discount |
| C | $9; 8 | 20 | 16 | -4 | -$56 | -$28 | $16 | -1.75 | Stop |

For example, Case A has promotion contribution `40 * ($9 - $7) = $80` and
baseline contribution `20 * ($10 - $7) = $60`, so incremental margin is $20.
The output's calculations were also checked across all 16,253 complete M5
episodes: incremental-unit differences were exact, and maximum absolute
floating-point discrepancies for incremental revenue, incremental margin, and
ROI were below `2.3e-12`.

## Three examples from the M5 output

These are representative complete one-week proxy episodes, selected to show
each recommendation. Dollar values are derived estimates under the 30% margin
assumption, not observed retailer costs.

| Item / store / week | Baseline → promo units | Incremental units | Incremental revenue | Incremental margin | ROI | Post-promo unit change | Action |
|---|---:|---:|---:|---:|---:|---:|---|
| HOUSEHOLD_1_428 / TX_2 / 2012-04-28 | 1 → 9 | +8 | $45.76 | $1.13 | 0.063 | +5 | Continue |
| FOODS_3_610 / TX_1 / 2015-01-10 | 6.5 → 9 | +2.5 | -$3.87 | -$7.34 | -0.832 | 0 | Reduce discount |
| HOUSEHOLD_2_001 / TX_1 / 2012-02-18 | 3.5 → 0 | -3.5 | -$22.65 | -$6.79 | unavailable | -4 | Stop |

ROI is unavailable for the last example because no units sold during its
promotion, so the discount-cost denominator is zero. The third-party M5 input
contains no observed unit costs or true promotion labels; all output is a
transparent screening result, not an automatic price instruction.
