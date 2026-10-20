"""Compare top FP-Growth pairs from a deterministic Instacart sample to candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from scipy.sparse import coo_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "module5"
SAMPLE_SIZE = 100_000
MIN_SUPPORT = 0.0025
MIN_LIFT = 1.5
TOP_N = 50


def build_sample_top_pairs(sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Run FP-Growth on all products in a deterministic prior-order sample."""
    prior_path = str(INPUT_DIR / "order_products__prior.csv")
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET memory_limit='768MB'")
        connection.execute("SET threads=2")
        connection.execute(
            """
            CREATE TEMP TABLE sampled_orders AS
            SELECT order_id
            FROM (SELECT DISTINCT order_id FROM read_csv_auto(?))
            ORDER BY hash(order_id)
            LIMIT ?
            """,
            [prior_path, sample_size],
        )
        sample_order_ids = np.asarray(
            [
                row[0]
                for row in connection.execute(
                    "SELECT order_id FROM sampled_orders ORDER BY order_id"
                ).fetchall()
            ],
            dtype=np.int64,
        )
        if len(sample_order_ids) != sample_size:
            raise ValueError(
                f"Requested {sample_size:,} orders, found {len(sample_order_ids):,}"
            )
        transactions = connection.execute(
            """
            SELECT DISTINCT sampled.order_id, items.product_id
            FROM read_csv_auto(?) AS items
            INNER JOIN sampled_orders AS sampled USING (order_id)
            """,
            [prior_path],
        ).df()

    observed_product_count = transactions["product_id"].nunique()
    product_order_counts = transactions.groupby("product_id")["order_id"].nunique()
    frequent_product_ids = product_order_counts.index[
        product_order_counts.ge(int(np.ceil(MIN_SUPPORT * sample_size)))
    ]
    transactions = transactions.loc[
        transactions["product_id"].isin(frequent_product_ids)
    ].copy()
    if transactions.empty:
        raise ValueError(
            "No products meet the sample minimum-support threshold; "
            "cannot mine frequent pairs"
        )

    order_positions = pd.Series(
        np.arange(sample_size, dtype=np.int32), index=sample_order_ids
    )
    product_ids = np.sort(transactions["product_id"].unique())
    product_positions = pd.Series(
        np.arange(len(product_ids), dtype=np.int32), index=product_ids
    )
    row_indices = order_positions.loc[transactions["order_id"].to_numpy()].to_numpy()
    column_indices = product_positions.loc[
        transactions["product_id"].to_numpy()
    ].to_numpy()
    basket_matrix = coo_matrix(
        (
            np.ones(len(transactions), dtype=np.bool_),
            (row_indices, column_indices),
        ),
        shape=(sample_size, len(product_ids)),
        dtype=np.bool_,
    ).tocsr()
    baskets = pd.DataFrame.sparse.from_spmatrix(
        basket_matrix,
        columns=[str(int(product_id)) for product_id in product_ids],
    )
    baskets = baskets.astype(pd.SparseDtype(np.bool_, fill_value=False))
    itemsets = fpgrowth(
        baskets,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=2,
    )
    rules = association_rules(itemsets, metric="lift", min_threshold=MIN_LIFT)
    pair_rules = rules.loc[
        rules["antecedents"].map(len).eq(1)
        & rules["consequents"].map(len).eq(1)
    ].copy()

    pair_rules["item_a"] = pair_rules["antecedents"].map(
        lambda values: int(next(iter(values)))
    )
    pair_rules["item_b"] = pair_rules["consequents"].map(
        lambda values: int(next(iter(values)))
    )
    pair_rules["pair"] = pair_rules.apply(
        lambda row: tuple(sorted((row["item_a"], row["item_b"]))), axis=1
    )
    pairs = (
        pair_rules.groupby("pair", as_index=False)
        .agg(
            support=("support", "max"),
            max_directional_confidence=("confidence", "max"),
            lift=("lift", "max"),
        )
    )
    pairs[["item_a", "item_b"]] = pd.DataFrame(
        pairs["pair"].tolist(), index=pairs.index
    )
    top_pairs = pairs.drop(columns="pair").sort_values(
        ["lift", "max_directional_confidence", "support", "item_a", "item_b"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).head(TOP_N).reset_index(drop=True)
    top_pairs.attrs["observed_product_count"] = int(observed_product_count)
    top_pairs.attrs["retained_product_count"] = int(len(product_ids))
    return top_pairs


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
        f"Products retained after lossless support pruning: "
        f"{sample_top.attrs['retained_product_count']:,} / "
        f"{sample_top.attrs['observed_product_count']:,}\n"
        f"Lift filter: > {MIN_LIFT}\n"
        f"Top distinct pairs: {len(sample_top)}\n"
        f"Pairs also in bundle_candidates.csv: {overlap_count} / {len(sample_top)}"
    )


if __name__ == "__main__":
    main()
