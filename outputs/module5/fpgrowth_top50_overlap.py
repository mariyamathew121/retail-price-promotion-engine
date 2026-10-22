"""Compare top FP-Growth pairs from a deterministic Instacart sample to candidates.

The FP-Growth mining itself lives in `retail_engine.module5.mine_fpgrowth_rules`;
this script collapses its directional rules to unordered pairs and checks how many
of the top pairs also appear in the full-data `bundle_candidates.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_engine.module5 import mine_fpgrowth_rules

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "module5"
SAMPLE_SIZE = 100_000
MIN_SUPPORT = 0.0025
MIN_LIFT = 1.5
TOP_N = 50


def build_sample_top_pairs(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Return the top distinct pairs from FP-Growth rules on the order sample."""
    rules = mine_fpgrowth_rules(INPUT_DIR, sample_size, MIN_SUPPORT, MIN_LIFT)
    if rules.empty:
        raise ValueError("FP-Growth produced no rules above the lift threshold")
    pairs = rules.assign(
        item_a=rules[["antecedent_id", "consequent_id"]].min(axis=1),
        item_b=rules[["antecedent_id", "consequent_id"]].max(axis=1),
    )
    pairs = pairs.groupby(["item_a", "item_b"], as_index=False).agg(
        support=("support", "max"),
        max_directional_confidence=("confidence", "max"),
        lift=("lift", "max"),
    )
    return (
        pairs.sort_values(
            ["lift", "max_directional_confidence", "support", "item_a", "item_b"],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(OUTPUT_DIR / "bundle_candidates.csv")
    sample_top = build_sample_top_pairs(args.sample_size)
    candidate_pairs = {
        tuple(sorted((int(row.item_a), int(row.item_b))))
        for row in candidates[["item_a", "item_b"]].itertuples(index=False)
    }
    sample_top["in_bundle_candidates"] = [
        tuple(sorted((int(item_a), int(item_b)))) in candidate_pairs
        for item_a, item_b in sample_top[["item_a", "item_b"]].itertuples(
            index=False, name=None
        )
    ]
    sample_top.to_csv(OUTPUT_DIR / "fpgrowth_top50_overlap.csv", index=False)
    overlap_count = int(sample_top["in_bundle_candidates"].sum())
    print(
        f"Sample orders: {args.sample_size:,}\n"
        f"FP-Growth min_support: {MIN_SUPPORT:.2%}; max_len: 2\n"
        f"Lift filter: > {MIN_LIFT}\n"
        f"Top distinct pairs: {len(sample_top)}\n"
        f"Pairs also in bundle_candidates.csv: {overlap_count} / {len(sample_top)}"
    )


if __name__ == "__main__":
    main()
