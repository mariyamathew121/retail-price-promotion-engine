"""Compare the same item's price elasticity across stores (per-SKU DML results)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from retail_engine.module2 import classify_elasticity

STABILITY_COLUMNS = [
    "item_id",
    "store_count",
    "estimated_store_count",
    "elasticity_min",
    "elasticity_median",
    "elasticity_max",
    "elasticity_spread",
    "sign_agreement_share",
    "modal_demand_type",
    "demand_type_agreement_share",
    "common_interval_exists",
    "stores",
]
DEFAULT_INPUT = Path("data/processed/module2/price_elasticities.csv")
DEFAULT_OUTPUT = Path("data/processed/module2/same_item_store_stability.csv")


def same_item_stability(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise each item that is estimated in two or more stores.

    Only fitted series are used. `common_interval_exists` is true when every
    store's 95% interval contains one shared value, i.e. the largest lower
    bound is at or below the smallest upper bound. Sign agreement is the share
    of stores with the same sign as the median estimate.
    """
    fitted = results.loc[results["fit_status"] == "estimated"].copy()
    fitted["demand_type"] = [
        classify_elasticity(e, lo, hi)[0]
        for e, lo, hi in zip(
            fitted["elasticity"], fitted["ci_lower"], fitted["ci_upper"], strict=True
        )
    ]
    rows = []
    for item_id, group in fitted.groupby("item_id", sort=True):
        if group["store_id"].nunique() < 2:
            continue
        estimates = group["elasticity"].to_numpy(dtype=float)
        median = float(np.median(estimates))
        same_sign = np.sign(estimates) == np.sign(median)
        types = group["demand_type"]
        modal_type = types.value_counts().index[0]
        rows.append(
            {
                "item_id": item_id,
                "store_count": int(group["store_id"].nunique()),
                "estimated_store_count": int(len(group)),
                "elasticity_min": float(estimates.min()),
                "elasticity_median": median,
                "elasticity_max": float(estimates.max()),
                "elasticity_spread": float(estimates.max() - estimates.min()),
                "sign_agreement_share": float(same_sign.mean()),
                "modal_demand_type": modal_type,
                "demand_type_agreement_share": float((types == modal_type).mean()),
                "common_interval_exists": bool(
                    group["ci_lower"].max() <= group["ci_upper"].min()
                ),
                "stores": " ".join(sorted(group["store_id"].astype(str))),
            }
        )
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results = pd.read_csv(args.input)
    stability = same_item_stability(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stability.to_csv(args.output, index=False)
    print(f"Items compared across stores: {len(stability):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
