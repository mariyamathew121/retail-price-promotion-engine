# Module 1 filtered-query benchmark

- Query filter: `item_id = FOODS_1_001`, `store_id = CA_1`, dates from
  `2015-01-01` inclusive to `2015-04-01` exclusive.
- Result: 90 item-store-day rows and 58 units sold.
- Elapsed time: 0.7209 seconds for one DuckDB query against
  `data/processed/clean_retail_data.parquet` on the current workspace.

This is a single observed run, not a benchmark across machines or repeated
cache states.
