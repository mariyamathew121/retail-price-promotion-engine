# Module 2 — Price elasticity interpretation

This analysis compares estimates from 100 selected M5 item-store series. The
median naive OLS elasticity was **−0.043**. When the inferred promotion-week
control was added, the median controlled OLS estimate moved to **−0.742**.
This change suggests the uncontrolled model does not separate price changes
from the demand conditions associated with promotion weeks.

Adding a linear time trend changed the median estimate to **+1.15**. The sign
change shows that the result depends on model specification; it should not be
read as evidence that raising prices increases demand. Across the individual
series, **44 of 100 DML point estimates** were positive, including **24**
significantly positive estimates at the reported 95% interval. Taken
together, these results make the per-series estimates too uncertain to use
as direct SKU price instructions.

The pooled fixed-effects PPML model estimated negative elasticities in all
seven departments, mostly between **−0.35 and −1.3**. It retains zero-sales
weeks and accounts for persistent item-store differences and store-week
shocks, but remains observational and does not produce SKU-specific
elasticities. The practical conclusion is to use the department-level results
to **prioritize controlled pricing experiments**, not to apply automatic
SKU-level actions.

For exact estimates, intervals, and model diagnostics, see
[`naive_dml_pooled_comparison.md`](./naive_dml_pooled_comparison.md),
[`pooled_model_scale_and_store_diagnostics.md`](./pooled_model_scale_and_store_diagnostics.md),
and [`../../assumptions.md`](../../assumptions.md).
