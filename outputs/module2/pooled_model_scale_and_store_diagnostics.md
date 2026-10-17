# Pooled price elasticity: scale and store diagnostics

Run with the 2,000 and 10,000 eligible item-store subsets using
`estimate-pooled-elasticity`. The 10,000-series run also used `--by-store`.
Both sample sizes are saved separately under
`data/processed/module2_pooled_2k/` and `data/processed/module2_pooled_10k/`.
The generated CSV files and audit JSONs contain the row counts, selected
series, exact estimates, intervals, and specification details.

The run used 2.84 GiB available RAM before the corrected 10,000-series
department comparison and 2.70 GiB before the 10,000-series by-store run
(15.68 GiB total RAM). Both completed; available RAM afterward was 2.67 GiB.
Peak process memory was not separately profiled.

## Models

- **FEOLS:** `log(units_sold) ~ log(price)`, positive-sales weeks only.
- **PPML:** units sold on `log(price)`, retaining zero-sales weeks.
- Both specifications include item-store and store-week fixed effects, with
  standard errors clustered by item-store. PyFixest drops singleton fixed
  effects; PPML may also remove observations due to separation.

## Department estimates at 10,000 series

| Department | FEOLS elasticity (95% CI) | PPML elasticity (95% CI) |
|---|---:|---:|
| FOODS_1 | -0.826 [-1.005, -0.647] | -1.301 [-1.597, -1.006] |
| FOODS_2 | -0.268 [-0.416, -0.121] | -1.058 [-1.312, -0.804] |
| FOODS_3 | -0.626 [-0.790, -0.462] | -0.883 [-1.048, -0.717] |
| HOBBIES_1 | -0.515 [-0.686, -0.345] | -0.680 [-0.993, -0.366] |
| HOBBIES_2 | -0.389 [-0.486, -0.292] | -0.719 [-0.911, -0.527] |
| HOUSEHOLD_1 | -0.335 [-0.464, -0.207] | -0.458 [-0.654, -0.262] |
| HOUSEHOLD_2 | -0.332 [-0.417, -0.246] | -0.353 [-0.512, -0.195] |

## Scaling comparison

| Department | PPML 2k | PPML 10k | PPML CI width 2k → 10k | FEOLS 2k | FEOLS 10k | FEOLS CI width 2k → 10k |
|---|---:|---:|---:|---:|---:|---:|
| FOODS_1 | -1.827 | -1.301 | 1.582 → 0.591 | -0.340 | -0.826 | 1.574 → 0.358 |
| FOODS_2 | -1.271 | -1.058 | 0.907 → 0.508 | -0.338 | -0.268 | 0.783 → 0.295 |
| FOODS_3 | -0.822 | -0.883 | 0.590 → 0.332 | -0.159 | -0.626 | 0.983 → 0.328 |
| HOBBIES_1 | -0.726 | -0.680 | 1.091 → 0.626 | -0.629 | -0.515 | 1.289 → 0.341 |
| HOBBIES_2 | -0.862 | -0.719 | 0.521 → 0.384 | -0.555 | -0.389 | 0.297 → 0.194 |
| HOUSEHOLD_1 | -0.139 | -0.458 | 1.060 → 0.393 | +0.275 | -0.335 | 0.983 → 0.257 |
| HOUSEHOLD_2 | -0.056 | -0.353 | 0.775 → 0.317 | -0.278 | -0.332 | 0.603 → 0.170 |

Median interval width narrowed by **56.7% for PPML** (0.907 to 0.393) and
**69.9% for FEOLS** (0.983 to 0.295). PPML estimates remained negative in all
departments, though several magnitudes moved. FEOLS HOUSEHOLD_1 changed sign.
Thus, the intervals tightened with the larger sample, but exact estimates
should not be described as fully stable across sample sizes/specifications.

The 10,000 selected-series counts are 1,505 FOODS_1; 1,580 FOODS_2; 1,580
FOODS_3; 1,580 HOBBIES_1; 596 HOBBIES_2; 1,580 HOUSEHOLD_1; and 1,579
HOUSEHOLD_2. HOBBIES_2 is capped by its 596 eligible series.

## Store-level stability

The 10,000-series by-store run estimated 70 department-by-store PPML models
(10 stores × 7 departments). The table reports the median and range of store
elasticity point estimates within each department, plus the number of stores
whose 95% interval excludes zero and the number with a positive point
estimate.

| Department | Median store elasticity | Range | 95% CIs exclude zero | Positive estimates |
|---|---:|---:|---:|---:|
| FOODS_1 | -2.005 | -3.543 to +1.551 | 10/10 | 1/10 |
| FOODS_2 | -1.376 | -1.843 to +0.940 | 9/10 | 1/10 |
| FOODS_3 | -1.378 | -2.231 to +0.569 | 9/10 | 1/10 |
| HOBBIES_1 | -1.269 | -2.703 to +0.936 | 6/10 | 1/10 |
| HOBBIES_2 | -0.930 | -1.276 to -0.287 | 8/10 | 0/10 |
| HOUSEHOLD_1 | -0.887 | -1.491 to +1.143 | 9/10 | 1/10 |
| HOUSEHOLD_2 | -0.473 | -0.876 to +0.879 | 6/10 | 1/10 |

There is substantial store heterogeneity. Each store-by-department model has
fewer clusters than the department-wide model; selected series range from 44
to 187 in individual store/departments. Interpret extremes cautiously.

## SKU default policy check

The output `item_store_pricing_defaults.csv` applies the department PPML class
and recommendation to the selected item-store series. The individual DML
estimate is included only as a class-disagreement flag. On this run, the pooled
default assigned 3,085 series to a controlled discount/promotion test and
6,915 to price protection/small-increase testing. Only 8 of the 100 matched
individual DML classes agreed with the pooled class; the remaining 92 are
review flags, not overrides. Treat the high disagreement as a model
validation warning, not an endorsement of automatic price actions.
