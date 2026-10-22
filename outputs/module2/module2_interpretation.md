# Module 2 interpretation

## Per-series elasticity

The per-series run estimated all 100 selected M5 item-store combinations. The
median uncontrolled log-log OLS elasticity is **-0.043**, the median controlled
OLS elasticity is **-0.742**, and the median cross-fitted DML elasticity is
**-0.484**. The median absolute DML-vs-controlled-OLS difference is **0.370**
with correlation **0.90**. Uncontrolled OLS omits the calendar and inferred
promotion controls; controlled OLS includes them linearly; DML uses flexible
nuisance models for the same observed controls.

The sign movement is a specification warning, not proof of bias removal.
Moving from uncontrolled OLS (**-0.043**) to seasonality, SNAP and event
controls changes little (**-0.054**); adding the inferred promotion flag moves
the median to **-0.742**. That flag is derived from price itself (a drop of at
least 10% below the prior eight-week maximum), so it removes much of the
deep-discount variation from the treatment: the controlled estimate leans on
smaller price changes and is not a clean "promotion-adjusted" elasticity. Adding
a linear time trend after that moves the median to **+1.145**, which is not
evidence that raising prices increases demand; it shows the result depends on
specification. These four medians are reproduced by `module2-control-ladder`
and recorded in [`control_ladder.md`](./control_ladder.md).

Classification now uses the same upper-confidence-bound rule as the pooled
model. A series is **price-sensitive** only when the upper 95% DML interval is
below -1, **less sensitive** when the upper bound is below 0 but not below -1,
and **unclear** otherwise. The current 100 per-series classifications are:
**26 price-sensitive**, **11 less sensitive**, and **63 unclear**. There are
**44 positive DML point estimates**, including **24 significantly positive**
intervals; these are flagged for review rather than action. The median DML
interval width is **5.743**, so per-series actions should be treated as test
hypotheses, not direct price instructions.

Individual review notes for positive point estimates are stored in the
`positive_elasticity_explanation` column of
`data/processed/module2/price_elasticities.csv`. The separate positive-cases
report plots the five largest examples for visual inspection, but the row-level
explanations cover the broader positive-estimate set.

## Pooled 10k results

The pooled fixed-effects PPML model is the default comparison because it
retains zero-sales weeks and borrows price variation across many products while
including item-store and store-week fixed effects. The 10,000-series run
estimates negative PPML elasticities in all seven departments:

| Department | Selected series | PPML elasticity | 95% CI | Default class |
|---|---:|---:|---:|---|
| FOODS_1 | 1,505 | -1.301 | [-1.597, -1.006] | Price-sensitive |
| FOODS_2 | 1,580 | -1.058 | [-1.312, -0.804] | Less sensitive |
| FOODS_3 | 1,580 | -0.881 | [-1.046, -0.715] | Less sensitive |
| HOBBIES_1 | 1,580 | -0.680 | [-0.993, -0.366] | Less sensitive |
| HOBBIES_2 | 596 | -0.719 | [-0.911, -0.527] | Less sensitive |
| HOUSEHOLD_1 | 1,580 | -0.458 | [-0.654, -0.262] | Less sensitive |
| HOUSEHOLD_2 | 1,579 | -0.353 | [-0.512, -0.195] | Less sensitive |

Only FOODS_1 is price-sensitive under the upper-bound rule. The selected
item-store defaults assign **1,505** series to controlled discount/promotion
tests and **8,495** to price protection or small-increase testing. These are
department defaults attached to item-store rows, not SKU-specific elasticities.
Among the 100 rows with both pooled defaults and per-series DML estimates,
**88** have a class disagreement (62 of the 63 per-series "unclear" rows are
"less sensitive" in the pooled default); that is a review flag, not an
override.

Category medians across department PPML point estimates are **-1.058 for
FOODS**, **-0.699 for HOBBIES**, and **-0.406 for HOUSEHOLD**. Store-level
pooled estimates show material heterogeneity, so a department default should
still be validated before use in a commercial price action.

## Same-item store stability

The same-item diagnostic compares per-series DML estimates for items estimated
in multiple stores. Example: `FOODS_3_090` is present in all 10 stores, with
elasticities from **+3.856** to **+14.294**, median **+6.292**, and all stores
classified **Unclear**. This is a useful warning: cross-store consistency in
sign does not make the positive estimate a normal demand response, especially
when the result remains observational and promotion exposure is inferred.

The full table is
`data/processed/module2/same_item_store_stability.csv`.

## DoubleML reference check

The C1 partialling-out replication used 10,000 observations and 595 features.
All four nuisance-learner specifications returned negative estimates with 95%
intervals excluding zero: linear **-1.756**, LassoCV **-1.767**, random forest
**-1.829**, and histogram gradient boosting **-1.813**. For one selected M5
series, `FOODS_1_096` at `WI_2`, the custom consecutive-fold estimate was
**-9.610** versus **-8.594** from `DoubleMLPLR`. This is an implementation
cross-check, not evidence that the M5 estimates are causally identified.

## Bottom line

Module 2 is useful for experiment prioritization. It is not sufficient for
automatic price changes because price movement is observational, promotion
exposure is inferred, confidence intervals are wide for many item-store
series, and pooled defaults are department-level rather than SKU-specific.
