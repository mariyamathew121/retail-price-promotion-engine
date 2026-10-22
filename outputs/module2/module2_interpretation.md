# Module 2: what the price elasticity results show

These results come from 100 selected M5 item-store series. They are observational
estimates, so they point to where a controlled price test is worth running. They
do not justify changing prices directly.

## 1. Three ways of estimating the same elasticity

| Method | Median elasticity |
|---|---:|
| Naive log-log OLS (no controls) | **-0.043** |
| Controlled OLS (season, SNAP, events, promotion flag) | **-0.742** |
| Cross-fitted DML (same controls, flexible models) | **-0.484** |

The naive estimate is close to zero because it does not separate price changes
from the other things happening in the same weeks. Adding the controls moves the
estimate further negative. Most of that move comes from the promotion flag.

The promotion flag is calculated from price itself: a drop of at least 10% below
the prior eight-week maximum. Controlling for it removes much of the deep-discount
price variation, so the controlled estimate depends on the smaller price changes
that remain. It should not be read as a clean estimate of price response.

Adding a linear time trend changes the median to **+1.145**. A positive sign
here does not mean higher prices increase demand. It shows the result depends on
the model specification. The estimate is only as reliable as the specification
behind it. The values above are reproduced by `module2-control-ladder`
([`control_ladder.md`](./control_ladder.md)).

DML and controlled OLS agree closely: median absolute difference **0.370**,
correlation **0.90**.

## 2. Positive estimates

**44 of the 100** DML point estimates are positive, and **24** of those are
significantly positive at the 95% level. A positive elasticity is not a normal
demand response, so these series are flagged for review and receive no pricing
action. Each one has a note in the `positive_elasticity_explanation` column of
`data/processed/module2/price_elasticities.csv`.

## 3. Classifications

Each series is classified using the upper bound of its 95% interval:

- **Price-sensitive:** upper bound below -1.
- **Less sensitive:** upper bound below 0 but not below -1.
- **Unclear:** anything else, which gets no action.

Result: **26 price-sensitive, 11 less sensitive, 63 unclear.** The median interval
is 5.743 wide, so every per-series classification is a test hypothesis rather
than an instruction.

## 4. Daily-grain check

The brief lists weekday, month, store and category as controls. The main models
are weekly because M5 prices change weekly, so weekday cannot enter them. To test
the brief's list, the same 100 series were re-estimated on daily rows with weekday
and month indicators added
([`daily_grain_check.md`](./daily_grain_check.md)).

- The two grains agree closely: correlation **0.91**, same sign for **94%** of
  series, same demand type for **92%**, and **99%** of intervals overlap.
- The daily run still has **48 positive** point estimates, **27** of them
  significantly positive, so the positive estimates are not caused by weekly
  aggregation.
- The medians differ: **-0.116 daily** against **-0.484 weekly**. Series volume
  does not explain the gap (Spearman correlation -0.17). I have not established
  its cause, so neither median should be taken as the typical elasticity.
- Daily rows repeat each weekly price seven times and add no new price variation.
  The daily intervals are therefore not more precise than the weekly ones.

## 5. Pooled department estimates

The pooled model (PPML with item-store and store-week fixed effects, run on
10,000 series) gives one estimate per department:

| Department | Series | PPML elasticity | 95% CI | Default class |
|---|---:|---:|---|---|
| FOODS_1 | 1,505 | -1.301 | [-1.597, -1.006] | Price-sensitive |
| FOODS_2 | 1,580 | -1.058 | [-1.312, -0.804] | Less sensitive |
| FOODS_3 | 1,580 | -0.881 | [-1.046, -0.715] | Less sensitive |
| HOBBIES_1 | 1,580 | -0.680 | [-0.993, -0.366] | Less sensitive |
| HOBBIES_2 | 596 | -0.719 | [-0.911, -0.527] | Less sensitive |
| HOUSEHOLD_1 | 1,580 | -0.458 | [-0.654, -0.262] | Less sensitive |
| HOUSEHOLD_2 | 1,579 | -0.353 | [-0.512, -0.195] | Less sensitive |

All seven are negative. Only FOODS_1 is price-sensitive under the upper-bound rule.
These are department defaults attached to item-store rows, not SKU-specific
elasticities. The pooled and per-series classes disagree for **88 of the 100**
series. That is a reason to review, not a rule that overrides either result.

## 6. Same-item stability

`FOODS_3_090` is estimated in 10 stores. Its elasticities range from **+3.856**
to **+14.294** (median **+6.292**), and every store is classified unclear. A
consistent positive sign across stores is still not a normal demand response.
The full table is in `data/processed/module2/same_item_store_stability.csv`.

## 7. C1 reference checks

The C1 tutorial data (10,000 rows, 595 features) was replicated with four
nuisance-learner choices. Every one returns a negative elasticity whose interval
excludes zero: linear **-1.756**, LassoCV **-1.767**, random forest **-1.829**,
histogram gradient boosting **-1.813**.

The C1 notebook is a tutorial with most cells left as `TODO` and no stored
outputs. Its only stated result is "around -1.8", which these replications match.
The replication uses the same method and settings: partialling-out PLR, five
folds, seed 123.

The notebook's sensitivity analysis, run with XGBoost learners, gives an
elasticity of **-1.798** (95% CI [-1.882, -1.713]). Assuming hidden confounders
that explain 3% of the residual variance in both outcome and price (the DoubleML
default), the estimate stays between **-1.917 and -1.678**. The robustness value
is **36.5%**: a hidden confounder would have to explain more than a third of the
residual variance in both price and demand to bring the estimate to zero. The
three strongest predictors, used as benchmarks, move the estimate by at most
**0.024**.

These checks apply to the tutorial data only. They do not validate the M5
estimates.

For one M5 series (`FOODS_1_096` at `WI_2`), the project's own consecutive-fold
estimate was **-9.610**, against **-8.594** from `DoubleMLPLR`. The two use
different fold schemes and inference, so this is a point-estimate comparison.

## Bottom line

Module 2 identifies where a controlled price test is worth running. It does not
set prices. The estimates are observational, the promotion flag is inferred, most
intervals are wide, and the pooled defaults are department-level, not SKU-specific.
