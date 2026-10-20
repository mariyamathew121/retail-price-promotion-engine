# FP-Growth top-50 overlap check

Ran mlxtend FP-Growth with `max_len=2` on the deterministic 100,000-order
Instacart prior-order sample. To match the existing bundle shortlist, the
sample minimum support was 0.25% (at least 250 sampled orders), and rules were
retained only when lift was strictly greater than 1.5. Products below the
sample support threshold were pruned before mining; this cannot exclude a
pair meeting the same support threshold because a pair's support cannot exceed
either product's individual support.

FP-Growth found 50 distinct pairs above the lift threshold. **45 of the top 50
pairs by lift also appear in `data/processed/module5/bundle_candidates.csv`.**
Five sampled pairs do not appear in that full-prior candidate file. This
difference is expected to be possible because ranking and support are
calculated on the 100,000-order sample here, but on all prior orders in the
candidate file.

The full top-50 results, including support, maximum directional confidence,
lift, and candidate-match flag, are in
`data/processed/module5/fpgrowth_top50_overlap.csv`. Recreate them with:

```powershell
py -3.10 outputs/module5/fpgrowth_top50_overlap.py
```
