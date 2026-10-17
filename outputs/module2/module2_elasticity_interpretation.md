# Module 2 elasticity interpretation

## What the estimates say

For the selected 100 M5 item-store series, median elasticity was -0.043 in
naive OLS, -0.742 in controlled OLS, and -0.484 in cross-fitted DML. The
unadjusted estimate is close to zero; controlling for annual seasonality,
SNAP, event weeks, and the inferred promotion flag changes it substantially.
The control ladder shows the largest single shift when promotion is added
(median OLS -0.054 to -0.742). Adding a linear trend then moves the median to
+1.145, which is a warning that these observational estimates are sensitive to
specification, not evidence that demand truly rises with price.

There is also little independent price movement for many individual series.
Only a small share of products change price in a typical store-week, so
per-series estimates are noisy; 44 of the 100 DML point estimates are positive
and 24 are significantly positive at the reported 95% interval. The revised
median DML interval width is 5.743. Those results are not suitable for
automated SKU-level price changes.

## What pooling changes

The department model uses item-store fixed effects to remove persistent
series-level differences and store-week fixed effects to absorb shocks shared
by products in a store and week. It clusters uncertainty by item-store. At
10,000 selected series, PPML elasticities were negative in all seven
departments, from -0.353 in HOUSEHOLD_2 to -1.301 in FOODS_1, with 95%
intervals excluding zero. Unlike the `log1p` DML outcome, the PPML log-link
coefficient has the conventional log-demand/log-price elasticity
interpretation and PPML retains zero-sales weeks.

Scaling from 2,000 to 10,000 series narrowed median PPML interval width by
56.7%, but some department coefficients shifted materially. The store
breakdown is also heterogeneous: six of the 70 department-store PPML point
estimates were positive, and department ranges included both signs in six
departments. Therefore pooled estimates improve precision, but do not establish
that one elasticity applies uniformly across stores or prove causality.

## Decision rule and limits

For now, use the pooled department PPML estimate and its interval as the
default classification for selected item-store combinations. Only recommend a
test when the interval excludes zero; treat negative estimates below -1 as
price-sensitive and those between -1 and zero as less sensitive. The
per-series DML classification is a disagreement flag only. In the 10,000-series
output, 92 of the 100 matched DML classes differ from the pooled class, which
should prompt review rather than an override.

The estimates still rely on observational M5 data and an inferred promotion
proxy. Store-week fixed effects do not remove product-specific time-varying
confounding, such as inventory shortages or targeted price changes. Use the
results to prioritize controlled experiments, not to change regular prices
automatically.

## DoubleML reference and implementation checks

The C1 partialling-out replication used 10,000 observations and 595 features.
All four nuisance-learner specifications returned negative estimates with
95% intervals excluding zero: linear -1.756 ([-1.838, -1.675]), LassoCV
-1.767 ([-1.850, -1.684]), random forest -1.829 ([-1.914, -1.744]), and
histogram gradient boosting -1.813 ([-1.897, -1.729]). Learner-specific
outcome-prediction RMSE ranged from 1.091 to 1.127, and price-prediction RMSE
from 0.249 to 0.266. The similarity of estimates is reassuring for this
reference example, but is not evidence that the same assumptions hold in M5.

For the selected M5 series `FOODS_1_096` at `WI_2`, the custom consecutive-fold
estimate was -9.610 (HAC SE 1.457), compared with -8.594 (DoubleML SE 0.762);
the point-estimate difference was -1.016. The intervals and standard errors
are not directly comparable because the fold construction and inference
methods differ. This is a single-series implementation cross-check, not a
validation of the extreme series-level elasticity or its causal meaning.
