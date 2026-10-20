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

   The brief lists weekday, month, store and category as controls. The
   per-series model is weekly, so weekday does not apply; month is captured by
   annual sine/cosine seasonality; store and category are constant within a
   single item-store series. The pooled model absorbs store and item effects
   through fixed effects.
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
8. **Reference validation and limits:** the C1 partialling-out DoubleML
   exercise was run on its separate 10,000-row Online Retail example with
   linear, LassoCV, random-forest, and histogram-gradient-boosting nuisance
   learners. A selected M5 series was also cross-checked against
   `DoubleMLPLR`. These are implementation checks, not proof of causal
   identification or validation of the M5 estimates. The C1 dataset is
   different from M5; the M5 check covers one series, uses random folds and
   DoubleML's default IID inference versus consecutive folds and HAC
   uncertainty in the custom estimator, and should be interpreted as a
   point-estimate comparison only. See
   `outputs/module2/module2_elasticity_interpretation.md` for the results.
9. **Pooled fixed-effects diagnostic:** the pooled model selects up to the
   requested number of
   eligible item-store series, allocating the sample as evenly as possible
   across departments subject to department eligibility counts, taking the
   highest-unit series within each department. Eligibility requires 104
   observed weeks, 52 positive-sales weeks, three price levels, and at least
   5% max/min price range. Within each department, estimate two models with
   item-store and store-week fixed effects, clustering standard errors by
   item-store:
   `log(units_sold)` on `log(price)` using positive-sales weeks only, and PPML
   of units on `log(price)` retaining zero-sales weeks. PPML is the preferred
   pooled comparison because it retains zeros; the log-linear model is shown
   alongside it as a sensitivity check. PyFixest drops singleton fixed-effect
   observations. The sample is selected using sales volume, and the estimates
   remain observational; fixed effects do not by themselves establish causal
   effects or remove item-specific time-varying confounding.
10. **Store stability:** with `--by-store`, repeat both specifications within
    each store and department using the same item-store clusters and store-week
    fixed effects. These estimates are diagnostic comparisons; smaller
    department-by-store samples can have wide intervals and are not assumed
    interchangeable.
11. **SKU pricing default:** for the selected item-store series, use the
    department-level PPML estimate and its 95% interval to assign the default
    class and recommendation. This is a department-level default attached to
    selected item-store rows, not an SKU-specific elasticity estimate. Assign
    price-sensitive only when the department estimate's upper 95% confidence
    bound is below -1; assign less sensitive when that upper bound is below
    zero but not below -1; otherwise mark unclear and take no pricing action.
    The matched per-series DML estimate and class are only a disagreement flag;
    they do not override the pooled department-level default.

## Module 3 — promotion uplift

1. **Promotion windows:** use the Module 1 inferred `promo_flag`; a promotion
    is one or more consecutive Saturday-start M5 weeks flagged for the same
    item-store. This is a price-drop proxy, not a retailer-supplied promotion
    calendar.
2. **Baseline:** for each window, take the mean units sold across the four
    latest non-promotion weeks found within the preceding eight calendar weeks.
    Multiply that weekly mean by the promotion duration to estimate baseline
    sales. If four non-promotion weeks are not available, report the episode as
    insufficient baseline and do not make a promotion recommendation.
3. **Gross margin:** assume category-specific regular-price gross margins of
    25% for FOODS, 35% for HOUSEHOLD, and 40% for HOBBIES. These are business
    assumptions, not observed M5 costs or data. `--gross-margin-rate` explicitly
    overrides the category rate for every category. Estimated unit cost is
    `regular_price * (1 - margin_rate_used)` and remains fixed during the
    promotion. The output records the category and margin rate used for each
    episode.
4. **Incremental revenue and margin:** compare actual promotion-period revenue
    and contribution with the estimated no-promotion baseline. Baseline revenue
    is baseline units multiplied by the promotion weeks' regular prices.
    Promotion contribution uses actual promotional prices less the estimated
    fixed unit cost; baseline contribution is baseline revenue times the
    assumed gross-margin rate.
5. **Promotion ROI:** calculate as incremental margin divided by discount
    cost, where discount cost is the regular-price-to-promotion-price
    difference times actual promoted units. If discount cost is zero, ROI is
    unavailable rather than treated as zero.
6. **Recommendation:** stop a promotion when incremental units are zero or
    negative; when incremental units are positive but incremental margin is
    zero or negative, reduce the discount; continue only when both incremental
    units and incremental margin are positive. These are screening rules, not
    automatic pricing actions.
7. **Persistent price cut review:** when the next week's price is observed and
   remains at or below 105% of the final flagged promotion week's price, set
   `price_persists` and replace the Continue/Reduce/Stop recommendation with
   `Possible permanent price change - review`. This is a conservative flag
   against treating a lasting price cut as a temporary promotion; it does not
   prove that the retailer intended a permanent change. If the next week is
   unavailable, the flag is false.
8. **Post-promotion dip:** compare the two following calendar weeks with the
    baseline weekly unit rate. Report a dip only when both post-promotion weeks
    are present; a negative value means post-promotion sales were below
    baseline. The post-promotion result is diagnostic and does not alter the
    promotion recommendation.

## Module 4 — markdown dataset audit

1. **Source and synthetic status:** the D2 CSV is treated as a synthetic
   modelling example, not representative evidence of real retailer behavior.
   The audit records the source-file SHA-256, dimensions, missingness, numeric
   ranges, duplicate rows, and stage-to-stage movements in
   `outputs/module4/markdown_dataset_audit.md`.
2. **Duplicate handling:** the raw file is preserved. Exact duplicate rows may
   be removed from an analysis copy; `Product_ID` alone is not a safe key until
   repeated identifiers have been checked for conflicting values.
3. **Markdown interpretation:** the `Markdown_1`–`Markdown_4` values are
   provisionally treated as discount fractions because they fall between 0
   and 1. The file has no data dictionary or dates, and adjacent markdown
   values are not consistently increasing. Do not interpret M1–M4 as a
   chronological markdown path until stage meaning and timing are confirmed.
4. **Sales interpretation:** `Historical_Sales` and `Sales_After_M1` through
   `Sales_After_M4` are not assumed to be weekly or equal-duration measures.
   Their observation windows are undocumented, so stage-to-stage differences
   are exploratory associations rather than causal uplift.
5. **Target leakage:** `Optimal Discount` is a supplied target/benchmark. It
   must not be included as a model input when estimating or predicting a
   discount choice or its sales response. Its derivation and validity are not
   documented.
6. **Stock and profit limits:** the source does not document the measurement
   timing/unit of `Stock_Level`; sales after M4 exceed stock in many rows.
   The Module 4 optimizer therefore uses stock as opening inventory and uses
   the cost/salvage values listed below as scenario assumptions only.
7. **Scope:** the dataset alone cannot establish a time-based optimizer,
   clearance outcome, realized profit, or ROI. The Module 4 output is a
   conditional simulation under explicit assumptions, not a validated retail
   recommendation. Real deployment requires dated outcomes, documented stage
   durations, aligned inventory periods, and observed unit costs/margins.
8. **Exploratory sales response:** for the initial D2 summary only, pair
   `Markdown_i` with `Sales_After_Mi` and calculate relative difference from
   `Historical_Sales`. Remove exact duplicate rows from this analysis copy,
   retain the source unchanged, and group markdown values into five
   data-derived quantiles. This is descriptive, not causal; `Optimal Discount`
   is excluded to prevent target leakage.
9. **Predictive diagnostic:** compare a metadata-only Extra Trees model with
   the same model plus one matched markdown field, using a fixed 80/20
   product-level split and relative sales difference as the target. Exclude
   IDs, other stage outcomes/markdowns, `Optimal Discount`, and fields of
   unclear timing. Held-out scores measure prediction within this synthetic
   dataset only; they are not causal elasticities, ROI, or pricing guidance.
10. **Markdown simulation assumptions (scenario defaults, not D2 facts):**
    - The dataset does not document the measurement period for `Historical_Sales`
      or matched stage sales. The primary rerun assumes `Historical_Sales` and
      each matched `Sales_After_Mk` are units per week. Sensitivity scenarios
      divide both values by assumed periods of 13 or 26 weeks before fitting.
    - Unit cost is 60% of `Original_Price`; salvage value is 20% of unit cost.
    - The demand horizon is eight weeks.
    - Fit a separate ordinary least-squares linear demand curve for each
      deduplicated product using the four matched markdown/sales pairs plus
      `(0, Historical_Sales)` after dividing sales by the assumed measurement
      period. Floor negative predicted weekly demand at zero.
    - **Weeks-cover eligibility rule (scenario policy):** calculate
      `weeks_cover = Stock_Level / assumed_weekly_baseline_sales`. Only allow a
      markdown if this is strictly greater than the simulated horizon; hold at
      0% otherwise. Equality is not eligible. This conservative guard is a
      policy assumption, not a data-derived threshold.
    - Evaluate markdowns from 0% through 50% inclusive in 10-percentage-point
      increments. Multiply fitted weekly demand by the horizon, cap total
      sales at `Stock_Level`, then calculate scenario profit as sales revenue
      plus salvage proceeds on leftover opening stock minus cost of all
      opening stock. Ties choose the lower markdown.
    - Compare the selected markdown against a fixed 30% markdown using the
      same demand curve and economics. Include the source `Optimal Discount`
      only as a clearly labelled comparison reference; never use it in
      fitting, prediction, or optimization. Compare profit only for products
      whose selected markdown differs from 30%; do not report a portfolio-wide
      profit-gap headline.
    - Recompute picks for all nine combinations of unit cost at 50%, 60%, or
      70% of original price and salvage at 10%, 20%, or 30% of unit cost.
      This economic sensitivity uses the 13-week sales-period / 8-week horizon
      scenario and reports recommendation changes relative to its default
      60%/20% case. Separately evaluate all combinations of assumed historical
      sales periods (1, 13, 26 weeks) and horizons (4, 8, 16 weeks).
    - Remove exact duplicate rows from the analysis input; reject repeated
      `Product_ID` values with conflicting records. Preserve the raw CSV.
    These are explicit scenario assumptions for an educational simulation,
    not facts provided by D2 or validated operating guidance.

## Module 5 — Instacart market basket analysis

1. **Basket grain:** treat one `order_id` as one basket and a product as present
   at most once in that basket. `order_products__prior.csv` is the historical
   association-discovery source; `order_products__train.csv` is kept separate
   as a held-out order set for checking pair co-occurrence outside discovery.
2. **Interpretation:** support, confidence, lift, or other association scores
   describe co-occurrence only. They do not establish that a bundle,
   cross-sell, or promotion causes incremental purchases or profit.
3. **Support sensitivity and working shortlist:** compare prior-order support
   thresholds of 0.10%, 0.25%, and 0.50%; use 0.25% as the provisional
   downstream shortlist threshold because it reduces the lowest-threshold
   candidate set while retaining a substantial share of its prior pair-event
   mass. Preserve the 0.10% candidates for broader discovery. Final candidates
   must also have prior lift strictly greater than 1.5. Rank by prior lift,
   then maximum directional confidence, then prior support. Report the share
   whose train lift also stays strictly above 1.5 and the Pearson correlation
   between prior and train lift when both are defined.
4. **Validation limit:** `train` orders are held out by order, not by user.
   The train users also appear in the prior history, so validation measures
   pair repeatability for partly familiar customers, not generalization to
   entirely unseen users.
5. **FP-Growth cross-check:** use mlxtend FP-Growth on a deterministic sample
   of 100,000 prior orders to cross-check three high-ranked pair supports
   against direct counts. Limit its basket columns to the six products in
   those three pairs to keep memory bounded; retain all sampled orders in the
   support denominator.
6. **M5 department mapping (assumption):** map `household` to `HOUSEHOLD`;
   map `bakery`, `beverages`, `breakfast`, `bulk`, `canned goods`, `dairy
   eggs`, `deli`, `dry goods pasta`, `frozen`, `international`, `meat
   seafood`, `pantry`, `produce`, and `snacks` to `FOODS`. Leave `alcohol`,
   `babies`, `missing`, `other`, `personal care`, and `pets` unmapped. Assign
   a pair-level `m5_category` only when both products map to the same category;
   otherwise leave it blank. This is a working taxonomy crosswalk, not an
   official Instacart-to-M5 mapping.
7. **Cross-price caveat (P3):** “bought together” is not evidence that one
   product's price changes demand for the other. Association rules do not
   identify cross-price elasticity, complementarity, or causal bundle uplift.
   Require cross-price/causal validation before pricing or promotion action.
