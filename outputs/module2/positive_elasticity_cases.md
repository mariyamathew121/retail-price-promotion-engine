# Positive item-store elasticity diagnostics

The plots show weekly units and average price for the five largest positive item-store DML estimates among the 100 selected series. The vertical axes are separate (units and currency price); visual co-movement is descriptive, not causal evidence. These estimates are classified Unclear and are not price recommendations.

| Item / store | DML estimate (95% CI) | Price-change weeks | Diagnostic reason | Plot |
|---|---:|---:|---|---|
| FOODS_3_541 / CA_3 | 24.39 [11.51, 37.26] | 12 | Log price is correlated with time (r=-0.41) and there are only 12 price changes; time-varying confounding is a plausible explanation for the positive estimate. | `positive_FOODS_3_541_CA_3.png` |
| FOODS_3_635 / CA_3 | 23.16 [12.08, 34.24] | 13 | Log price is correlated with time (r=-0.41) and there are only 13 price changes; time-varying confounding is a plausible explanation for the positive estimate. | `positive_FOODS_3_635_CA_3.png` |
| FOODS_3_288 / CA_3 | 22.00 [19.21, 24.79] | 4 | Only 4 observed price changes across 173 weeks make the positive estimate sensitive to sparse price variation; do not interpret it as a demand response. | `positive_FOODS_3_288_CA_3.png` |
| HOUSEHOLD_1_303 / CA_3 | 17.81 [9.62, 26.00] | 4 | Only 4 observed price changes across 278 weeks make the positive estimate sensitive to sparse price variation; do not interpret it as a demand response. | `positive_HOUSEHOLD_1_303_CA_3.png` |
| FOODS_3_362 / CA_3 | 15.98 [12.71, 19.25] | 10 | The very large positive estimate (15.98) persists with 10 price changes and 5 price levels; investigate time-varying confounding, inventory, and price measurement rather than acting on it. | `positive_FOODS_3_362_CA_3.png` |

A positive estimate—even with an interval excluding zero—is not a plausible demand curve by itself. Possible explanations include endogenous pricing, stock availability, common time shocks, sparse price variation, and the use of `log1p(units_sold)` as the outcome.
