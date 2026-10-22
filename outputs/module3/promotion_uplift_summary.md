# Module 3 promotion uplift — first M5 run

## Method and outputs

The analysis reads `data/processed/clean_retail_data.parquet` and is produced
by `src/retail_engine/module3.py`. It treats consecutive flagged Saturday-start
weeks for an item-store as one inferred promotion episode. The baseline is the
mean unit sales from the four calendar weeks immediately before each episode,
requiring all four weeks to be observed and non-promotional. Assumed
regular-price gross margins are 25% for FOODS,
35% for HOUSEHOLD, and 40% for HOBBIES; estimated unit cost is held fixed
through the promotion. These are assumptions, not observed data. The table
records category, average discount depth, and the margin rate used for each
episode. See `assumptions.md` for the full definitions and recommendation
rules.

The generated, row-level table and its audit are
`data/processed/module3/promotion_uplift.csv` and
`data/processed/module3/promotion_uplift_audit.json`. Those reproducible data
artifacts are ignored by Git; this script and report document how they were
created.

The optional CausalML diagnostic has also been generated at
`data/processed/module3_causalml/promotion_tlearner_observational.csv`, with
its audit in the same folder. It fits a T-learner on 1,000 selected promoted
item-store series and reports 30 department/store summaries on a chronological
holdout. Because the treatment is still the inferred M5 price-drop flag, these
outputs are observational diagnostics, not causal uplift estimates or
promotion recommendations.

## Run totals

| Result | Count |
|---|---:|
| Inferred promotion episodes | 16,749 |
| Fully evaluated (four baseline weeks available) | 15,669 |
| Possible permanent price change — review | 6,980 (6,713 complete; 267 also lack baseline) |
| Marked for review (insufficient baseline, not persistent) | 813 |
| Continue promotion | 1,239 |
| Reduce discount | 3,276 |
| Stop promotion | 4,441 |

The persistence check flags an episode when the following observed week's
price is at most 105% of its final flagged-week price. This review warning
overrides the Continue/Reduce/Stop label; it also takes precedence over the
insufficient-baseline label when both apply. An insufficient baseline remains
visible in `analysis_status`. The current CSV and audit JSON were regenerated
with `estimate-promotion-uplift`.

Among complete episodes, median duration was five weeks; the 75th percentile
was eight weeks. These are runs in the Module 1 price-drop proxy, not verified
retailer campaigns. Of 15,669 complete episodes, 3,271 (20.9%) had a zero-unit
baseline, and 6,556 (41.8%) had promotion sales at or below baseline sales.
These sparse/weak-sales cases are frequent enough to affect segment medians;
in particular, median ROI values of -1.000 should not be read as evidence of a
generalizable campaign effect. Check sparse-history and inventory context
before acting. The two post-promotion weeks were available for 98.8% of
complete episodes.

## Post-promotion dip

Research question 4 asks whether customers stock up during a promotion and buy
less afterwards. The check compares units in the two weeks after each episode
with the baseline rate (`post_promo_dip_units`; negative means a dip). It is
summarised in `promotion_uplift_audit.json` under `post_promotion_dip_summary`
and is diagnostic only: it does not change the recommendation.

| Episodes (complete, both post weeks observed) | Count | Share below baseline | Median post vs baseline |
|---|---:|---:|---:|
| All | 15,475 | 35.8% | +5.3% |
| Excluding possible permanent price cuts | 8,769 | 37.3% | +6.8% |
| Possible permanent price cuts only | 6,706 | 33.8% | 0.0% |

Median relative change uses episodes with a positive baseline. **There is no
evidence of a typical post-promotion dip in this proxy data**: the median
episode sells slightly *more* than baseline afterwards, and only about a third
sit below it. A dip may still exist for individual SKUs, and a flat result is
also what weekly-grain data with sparse sales and a price-drop proxy for
promotions would produce, so this does not show that stockpiling never happens.

## Hand-calculated checks

`tests/test_module3.py::test_three_hand_calculated_promotion_examples` checks
the following arithmetic at a $10 regular price and explicit 30% margin
override (unit cost $7). Each case has four baseline weeks at 10 units/week,
two promotion weeks, and two post-promotion weeks at 8 units/week.

| Case | Promotion price; units/week | Baseline sales | Promo sales | Incremental units | Incremental revenue | Incremental margin | Discount cost | ROI | Action |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | $9; 20 | 20 | 40 | 20 | $160 | $20 | $40 | 0.50 | Continue |
| B | $8; 12 | 20 | 24 | 4 | -$8 | -$36 | $48 | -0.75 | Reduce discount |
| C | $9; 8 | 20 | 16 | -4 | -$56 | -$28 | $16 | -1.75 | Stop |

For example, Case A has promotion contribution `40 * ($9 - $7) = $80` and
baseline contribution `20 * ($10 - $7) = $60`, so incremental margin is $20.
The output's calculations were also checked across all 15,669 complete M5
episodes: incremental-unit differences were exact, and maximum absolute
floating-point discrepancies for incremental revenue and margin were below
`4.6e-12`.

### Example using the FOODS category margin

The category-default hand check uses a $10 regular price and the assumed
FOODS margin of 25%, so estimated unit cost is $7.50. Four non-promotion weeks
each sell 10 units; the promotion lasts two weeks at $9 and sells 20 units per
week.

- Baseline: 20 units; revenue `$200`; contribution `$200 * 0.25 = $50`.
- Promotion: 40 units; revenue `$360`; contribution `40 * ($9 - $7.50) = $60`.
- Incremental units: `40 - 20 = 20`.
- Incremental revenue: `$360 - $200 = $160`.
- Incremental margin: `$60 - $50 = $10`.
- Discount cost: `($10 - $9) * 40 = $40`; ROI: `$10 / $40 = 0.25`.
- Action: **Continue promotion**. `margin_rate_used` is `0.25`.

This is the default-margin check in
`tests/test_module3.py::test_category_margin_is_used_without_an_override`;
the three earlier arithmetic cases retain an explicit 30% margin override.

## P1: promotion ROI by segment

The requested segment calculation is saved verbatim in
`outputs/module3/roi_by_segment.txt`. It reports median promotion ROI after
excluding incomplete and possible-permanent-price-change rows. For the
remaining 8,956 complete episodes, the medians are:

| Segment | Median ROI |
|---|---:|
| Depth (10%, 15%] | -0.135 |
| Depth (15%, 25%] | -0.754 |
| Depth (25%, 40%] | -1.011 |
| Depth (40%, 100%] | -1.000 |
| FOODS | -0.945 |
| HOBBIES | -1.090 |
| HOUSEHOLD | -1.000 |

The store-level medians are CA_1 -0.689, CA_2 -0.508, CA_3 -1.545,
CA_4 -0.560, TX_1 -1.000, TX_2 -0.983, TX_3 -1.000, WI_1 -0.886,
WI_2 -0.798, and WI_3 -0.785. Every category, store, and discount-depth
band has a negative median ROI; the deeper discount bands have more negative
medians than the shallowest band. This is consistent with discount cost
eroding incremental contribution, but it does not establish causal effects:
promotions are inferred from price drops and costs/margins are assumptions.
Interpret the -1.000 medians alongside the data-quality rates above: 20.9% of
complete episodes have zero baseline units, and 41.8% have promotion sales no
higher than baseline. The sparse and non-incremental observations are
substantial and can make the segment summaries sensitive to low-volume rows.
The filtered recommendation counts are 1,239 Continue, 3,276 Reduce discount,
and 4,441 Stop.

## Three examples from the M5 output

These are representative complete proxy episodes, selected to show each
recommendation. Dollar values are derived estimates under the category margin
assumptions, not observed retailer costs.

| Item / store / week | Category; margin | Baseline → promo units | Incremental units | Incremental revenue | Incremental margin | ROI | Post-promo unit change | Action |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HOBBIES_1_243 / CA_4 / 2016-03-26 | HOBBIES; 40% | 0.5 → 1 | +0.5 | $5.03 | $0.87 | 0.453 | 0 | Continue |
| FOODS_3_098 / TX_2 / 2015-10-03 | FOODS; 25% | 4.75 → 5 | +0.25 | -$6.28 | -$7.12 | -0.962 | +8.5 | Reduce discount |
| FOODS_3_812 / TX_1 / 2012-12-22 | FOODS; 25% | 9 → 1 | -8 | -$23.43 | -$6.75 | -5.672 | +4 | Stop |

The third-party M5 input contains no observed unit costs or true promotion
labels; all output is a transparent screening result, not an automatic price
instruction.
