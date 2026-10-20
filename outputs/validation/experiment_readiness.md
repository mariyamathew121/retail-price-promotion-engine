# Experiment readiness and real-data contracts

The current M5, D2, and Instacart analyses are useful for screening and
planning. They cannot establish causal price, promotion, markdown, or bundle
effects. Use the data contracts below before describing an effect as causal or
using it to change retail prices.

## Module 2 — regular-price experiment

Randomize eligible item-store clusters to a pre-registered price policy and
control price; avoid assigning different arms to stores with substantial
customer or inventory spillover. Record the assignment before outcomes occur.

Required fields: experiment and assignment IDs, item/store, assigned arm,
planned and actual price, assignment/start/end timestamps, units sold,
in-stock/stockout status, promotion exposure, unit cost, and calendar/event
controls. Analyze intent-to-treat first, report uncertainty and compliance,
and predefine guardrails for revenue, contribution, and customer outcomes.
Use the observational M5 elasticity outputs only to choose candidate tests and
prioritize strata.

## Module 3 — promotion experiment

Randomly assign eligible item-store or store clusters to campaign versus
control, with a documented assignment unit and campaign schedule. Record
campaign IDs, planned and actual discount depth, treatment assignment,
exposure, item/store, daily sales, actual prices, stock availability, unit
cost/margin, and the post-campaign window. Preserve untreated controls and
avoid defining treatment from the observed price drop after the fact.

The new CausalML T-learner is an exploratory model on inferred exposure. It is
not a substitute for this design. Evaluate a randomized test with intent-to-
treat incremental units and contribution, confidence intervals, and
predefined post-promotion outcomes.

## Module 4 — real markdown evaluation

Do not apply the D2 simulator outputs to real stock. A valid operational input
needs one row per item-store-date (or a precisely defined weekly grain) with
SKU/store IDs, observation date, opening and closing stock, receipts,
stockouts, units sold, actual selling price/markdown, unit cost, realized
salvage/clearance value, and a documented season-end or decision horizon.
Define each sales measurement period and align sales, price, and inventory
periods. Hold out later periods or stores for validation; compare predicted
profit with realized profit and a predeclared fixed-markdown policy.

## Module 5 — bundle/cross-price experiment

Randomly assign eligible customers or stores to bundle/offer treatment and
control. Record assignment and exposure, both products' item IDs and prices,
standalone and bundle offer prices, order/customer IDs, quantities, net
revenue, unit costs, availability, substitutions, returns, and experiment
dates. Measure incremental contribution and cannibalization using the
randomized assignment, not just basket co-occurrence or lift.

Instacart has no prices or randomized offer assignment; its association output
can only shortlist hypotheses. The current category crosswalk is a taxonomy
assumption and should be checked against a retailer product master before use.

## Shared release checks

- Pre-register the randomization unit, primary outcome, exclusion rules,
  minimum detectable effect, power/sample-size target, and stop/guardrail rules.
- Keep raw source snapshots, data dictionary, assignment log, analysis code,
  and generated audit outputs together; record file hashes and run parameters.
- Check treatment/control overlap, missingness, stockout imbalance, data
  leakage, and confidence intervals before publishing an effect estimate.
- Separate descriptive associations, observational model estimates, and
  randomized causal estimates in reports and dashboards.
- Never auto-deploy pricing or promotion actions from the current prototype.
