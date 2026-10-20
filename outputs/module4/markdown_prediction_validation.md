# Module 4 — held-out predictive check

## Method and data boundary

- Source: `data/raw/markdown/SYNTHETIC Markdown Dataset.csv`
- SHA-256: `9e6fb89fb4637da2573773de8d1110e8eb3e59439d49f3b197ee2f41386ef000`
- Exact duplicate rows were removed; analysis rows: 43,400.
- Split: one reproducible 80/20 random product-level train/test split
  (`random_state=20261018`).
- Target: relative difference `(Sales_After_Mi - Historical_Sales) /
  Historical_Sales` for each stage.
- Learner: Extra Trees regression (`n_estimators=160`,
  `min_samples_leaf=15`, `max_features=0.9`).
- Compared models: metadata-only and metadata plus that stage's matched
  markdown field. The metadata is category, brand, season, original price,
  competitor price, seasonality factor, and historical sales.
- Deliberately excluded: `Product_ID`, all other stage markdown/sales fields,
  `Stock_Level`, `Customer Ratings`, `Return Rate`, `Promotion_Type`, and
  `Optimal Discount`. Fields with unclear measurement timing are not used as
  predictors; excluding `Optimal Discount` avoids target leakage.

## Test-set results

MAE is shown in percentage points of relative sales difference. Negative
improvement means the markdown-augmented model had a higher (worse) MAE.

| Stage | Metadata-only MAE (pp) | Plus markdown MAE (pp) | MAE improvement (pp) | Metadata-only R² | Plus markdown R² | R² improvement |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 6.47 | 4.34 | +2.12 | 0.527 | 0.791 | +0.264 |
| M2 | 11.88 | 10.46 | +1.42 | 0.703 | 0.775 | +0.072 |
| M3 | 14.79 | 14.25 | +0.54 | 0.763 | 0.786 | +0.023 |
| M4 | 22.32 | 21.68 | +0.64 | 0.812 | 0.824 | +0.011 |

Adding each stage's matched markdown feature improved held-out MAE on this fixed split.

This is one held-out split of a synthetic, cross-sectional dataset. Predictive
improvement does not estimate the causal impact of markdowns, validate the
supplied sales measurement windows, establish profitability, or justify a
discount recommendation. Use the result only as a modelling diagnostic; a
real decision model requires dated outcomes, documented stage windows,
aligned inventory, and unit cost/margin data.
