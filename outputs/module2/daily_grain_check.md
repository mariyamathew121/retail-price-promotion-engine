# Module 2 daily-grain check (brief's control list)

The primary models are weekly because M5 prices change weekly. This check re-estimates the same series on daily rows with weekday and month indicators, SNAP, event and the inferred promotion flag as controls (outcome `log1p(units)`, treatment `log(price)`, cross-fitted Extra Trees DML, Newey-West lag 28 days). Store and category are constant within an item-store series, so they cannot be controls here.

- Series estimated: 100 of 100
- Median daily DML elasticity: -0.116 (weekly: -0.484)
- Median absolute difference from the weekly estimate: 0.765
- Correlation with the weekly estimate: 0.91
- Same sign as the weekly estimate: 94%
- Same demand type as the weekly estimate: 92%
- Weekly and daily 95% intervals overlap: 99%
- Median interval width: daily 3.875, weekly 5.743
- Daily demand types: 63 Unclear, 27 Price-sensitive, 10 Less sensitive

Daily rows repeat each week's price seven times, so the extra rows add weekday and month detail to the outcome but no new price variation; the intervals use a 28-day Newey-West lag to account for that, and should not be read as more precise than the weekly ones. Observational estimates; see assumptions.md.
