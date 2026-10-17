# Assumptions and decision rules

This file records assumptions that affect generated outputs. Add a dated entry
here whenever a later module introduces a business threshold, margin, or model
choice.

## Module 1 — M5 data foundation

1. **M5 has no promotion indicator.** `promo_flag` is an inferred proxy, not a
   source-provided label. It is `1` only when `discount_depth` is at least 10%
   and that SKU-store has an observed price in the preceding eight calendar
   weeks. It is `0` when there is no prior price history.
2. **Regular-price estimate:** for each `store_id` and `item_id`, use the
   maximum observed `sell_price` among the previous eight calendar weeks,
   excluding the current week. Calendar weeks are anchored to their M5 calendar
   start date, and the prior window includes the week exactly eight weeks
   earlier. If there is no price history in that window, set `regular_price` to
   the current `sell_price`.
3. **Discount depth:** `max(0, (regular_price - price) / regular_price)`. A
   depth below 10% is not flagged as a promotion. This rolling historical
   comparison is a transparent price-drop proxy; it does not establish why the
   price was set or prove a retailer ran a promotion.
4. **No-price rows:** a row with no matching `store_id`, `item_id`, and
   `wm_yr_wk` sell-price record is treated as not active / not price-observable
   and excluded. The exact excluded row count is written to the audit JSON.
5. **Calendar event fields:** use event 1 when present; otherwise use event 2.
   The M5 SNAP indicator is selected from the column for the row's state
   (`snap_CA`, `snap_TX`, or `snap_WI`).
6. **Price grain:** M5 sell prices are weekly, so each day in a given M5 week
   receives that week's price and inferred flag.
7. **Spot-check sample:** the ten SKU-store histories are selected first from
   combinations with at least one threshold-qualified price drop, then filled
   with other combinations if fewer than ten have a flagged drop. The generated
   plot shows weekly observed prices and the prior-eight-week maximum.

These choices are suitable for a reproducible learning baseline, not causal
proof of promotion exposure. Any later causal or financial decision must
account for this measurement limitation.

## Module 2 — price elasticity

1. **Time grain:** aggregate daily sales to Saturday–Friday weeks, matching the
   weekly price grain in M5. Prices are averaged within the week; they should be
   constant for a SKU-store M5 week.
2. **Eligibility:** select up to 100 item-store combinations by total units
   sold, requiring at least 104 observed weeks, 52 positive-sales weeks, three
   distinct prices, and a maximum/minimum price range of at least 5%. These
   thresholds are explicit defaults and can be changed through the CLI.
3. **Zero sales:** retain every observed week, including weeks with zero units,
   and model `log1p(units_sold)` so zero-demand observations remain in the
   outcome. Week indexes retain original calendar positions for the HAC
   interval.
4. **DML specification:** use cross-fitted partialling-out DML with Extra Trees
   nuisance models to predict log price and `log1p` units from annual
   sine/cosine seasonality, SNAP exposure, event-week status, and the inferred
   promotion-week flag. Do not include a linear time trend. Use five
   consecutive folds and 50 trees by default. The point estimate is the
   coefficient of residualized `log1p` units on residualized log price.
5. **Baselines and uncertainty:** report uncontrolled log-price/log1p-units
   OLS (`naive_elasticity`) and controlled OLS with the DML controls
   (`ols_elasticity`). DML and both OLS models report 95% Newey-West intervals
   with a maximum lag of four calendar weeks. HAC lag pairs are matched by
   actual week distance, so gaps in the observed series are not treated as
   adjacent weeks. This is a practical time-series adjustment, not a guarantee
   of valid inference.
6. **Interpretation:** treat estimates as causal only if the observed controls
   adequately address confounding and there is independent price variation.
   Unobserved inventory constraints, price-setting decisions, and the inferred
   promotion proxy can bias the estimates. Recommendations are test hypotheses,
   not automatic price changes.
7. **Action threshold:** apply the supplied business rules only when the DML
   confidence interval excludes zero. Positive or statistically unclear
   estimates are marked unclear and receive no action.
8. **Reference status:** this is an M5-specific implementation of the
   partialling-out DML approach. It does not reproduce the C1 notebook
   cell-for-cell on the notebook's separate Online Retail example data, nor
   compare its full set of linear, Lasso, random-forest, and boosting nuisance
   learners or sensitivity analysis. Complete that reference exercise before
   treating the M5 estimates as a validated causal model.
9. **Pooled fixed-effects diagnostic:** the pooled model selects up to 2,000
   eligible item-store series, allocating the sample as evenly as possible
   across departments and taking the highest-unit series within each
   department. Eligibility requires 104 observed weeks, 52 positive-sales
   weeks, three price levels, and at least 5% max/min price range. Within each
   department, estimate `log(units_sold)` on `log(price)` with item-store and
   store-week fixed effects; cluster standard errors by item-store. Zero-sales
   weeks are excluded because the log outcome is undefined, so this estimates
   the relationship among positive-sales weeks and may be affected by
   selection. The store-week fixed effects absorb shocks shared by products in
   that store and week, but the model remains observational and does not by
   itself establish causal effects.
