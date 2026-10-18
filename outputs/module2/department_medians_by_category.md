# Pooled PPML department medians by category

Estimates are from the latest 10,000-series pooled run
(`data/processed/module2_pooled/pooled_price_elasticities.csv`). Category
median is the unweighted median across the listed department-level PPML point
estimates; it is not an SKU-level or category-wide causal elasticity.

| Category | Department PPML estimates | Median across departments |
|---|---|---:|
| FOODS | FOODS_1 -1.301; FOODS_2 -1.058; FOODS_3 -0.884 | **-1.058** |
| HOBBIES | HOBBIES_1 -0.680; HOBBIES_2 -0.719 | **-0.699** |
| HOUSEHOLD | HOUSEHOLD_1 -0.458; HOUSEHOLD_2 -0.353 | **-0.406** |

PPML uses item-store and store-week fixed effects and standard errors clustered
by item-store. The inputs are observational M5 data; interpret these pooled
category summaries as descriptive model outputs, not causal guarantees.
