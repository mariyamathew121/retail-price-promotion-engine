# Naive OLS, controlled DML, and pooled elasticity

The 100-series baselines tell very different stories depending on what is
controlled. Median naive log-log OLS elasticity is about **-0.043** (roughly
-0.05): without controls, the typical estimate is close to zero. The control
ladder shows the promotion-week control making the largest shift: the median
controlled OLS estimate moves from about **-0.05 to -0.74** when the inferred
promotion flag is added. This is consistent with discounts coinciding with
high-demand weeks; the naive association mixes price response with the
circumstances that caused the price to change. In the full controlled
specification, median OLS is **-0.742**, while the cross-fitted DML median is
**-0.484**. DML flexibly estimates the price and demand nuisance functions, but
it does not cure unmeasured confounding or weak price variation. Its outcome
is `log1p(units_sold)`, so its coefficient should be read cautiously as an
elasticity, especially for low/zero-demand series.

The ladder is a useful warning about specification sensitivity: adding a
linear time trend after promotion shifts the median to **+1.15**. That sign
flip is not evidence that raising price increases demand; it indicates that
price, demand, and time are entangled in the observational data. The trend was
therefore excluded from the primary per-series DML controls, but the ladder's
instability remains important evidence against individual automated actions.

Pooling borrows price variation across many products and absorbs persistent
item-store differences and shocks shared by a store-week. In the 10,000-series
run, PPML estimates range from **-1.301 (FOODS_1)** to **-0.353
(HOUSEHOLD_2)**; category medians across departments are **-1.058 for FOODS**,
**-0.699 for HOBBIES**, and **-0.406 for HOUSEHOLD**. PPML retains zero-sales
weeks and uses item-store-clustered uncertainty. These department-level
estimates are more precise than the per-series results, but remain
observational and heterogeneous by store. They are **department-level
defaults, not SKU-specific estimates**. By the current upper-CI rule, only
FOODS_1 is price-sensitive; FOODS_2's point estimate is below -1 but its upper
95% bound is -0.804, so it is classified less sensitive.

Taken together: naive OLS understates the apparent response, adding the
promotion control makes it substantially more negative, DML partially adjusts
the relationship using flexible controls, and pooled PPML gives more precise
department estimates by using cross-product price variation. None alone
proves causality. Treat these outputs as experiment-prioritization evidence,
not instructions to change prices automatically.

Sources: `data/processed/module2/price_elasticities_audit.json`,
the recorded control-ladder output from the Module 2 diagnostics, and
`data/processed/module2_pooled/pooled_price_elasticities.csv`.
