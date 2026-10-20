"""Build and validate Instacart bundle candidates from market-basket data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "module5"
DEFAULT_SUPPORT_THRESHOLD = 0.0025
DEFAULT_LIFT_THRESHOLD = 1.5
SUPPORT_SENSITIVITY = (0.001, 0.0025, 0.005)
FP_GROWTH_SAMPLE_SIZE = 100_000
VALIDATION_STATUS = "Needs cross-price validation"

REQUIRED_FILES = (
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)

DEPARTMENT_TO_M5_CATEGORY = {
    "alcohol": None,
    "babies": None,
    "bakery": "FOODS",
    "beverages": "FOODS",
    "breakfast": "FOODS",
    "bulk": "FOODS",
    "canned goods": "FOODS",
    "dairy eggs": "FOODS",
    "deli": "FOODS",
    "dry goods pasta": "FOODS",
    "frozen": "FOODS",
    "household": "HOUSEHOLD",
    "international": "FOODS",
    "meat seafood": "FOODS",
    "missing": None,
    "other": None,
    "pantry": "FOODS",
    "personal care": None,
    "pets": None,
    "produce": "FOODS",
    "snacks": "FOODS",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_input_files(input_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Instacart dataset is missing required files: {missing}")


def pair_statistics(
    pair_orders: int,
    item_a_orders: int,
    item_b_orders: int,
    total_orders: int,
) -> dict[str, float]:
    """Return basket support, directional confidence, and lift for a pair."""
    if total_orders <= 0:
        raise ValueError("total_orders must be positive")
    if pair_orders < 0 or item_a_orders <= 0 or item_b_orders <= 0:
        raise ValueError("Pair counts must be non-negative and item counts positive")
    if pair_orders > min(item_a_orders, item_b_orders) or max(
        item_a_orders, item_b_orders
    ) > total_orders:
        raise ValueError("Pair and item counts must be consistent with total_orders")
    support = pair_orders / total_orders
    confidence_a_to_b = pair_orders / item_a_orders
    confidence_b_to_a = pair_orders / item_b_orders
    lift = (pair_orders * total_orders) / (item_a_orders * item_b_orders)
    return {
        "support": support,
        "confidence_a_to_b": confidence_a_to_b,
        "confidence_b_to_a": confidence_b_to_a,
        "prior_lift": lift,
    }


def map_m5_category(department_a: str, department_b: str) -> str:
    """Map a pair only when both departments map to the same M5 category."""
    category_a = DEPARTMENT_TO_M5_CATEGORY.get(department_a.strip().lower())
    category_b = DEPARTMENT_TO_M5_CATEGORY.get(department_b.strip().lower())
    return category_a if category_a is not None and category_a == category_b else ""


def rank_bundle_candidates(
    pairs: pd.DataFrame,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    lift_threshold: float = DEFAULT_LIFT_THRESHOLD,
) -> pd.DataFrame:
    """Filter by prior support/lift and rank by lift, max directional confidence, support."""
    required = {
        "prior_support",
        "prior_lift",
        "confidence_a_to_b",
        "confidence_b_to_a",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Pair table is missing metric columns: {missing}")
    if not 0 <= support_threshold <= 1 or lift_threshold < 0:
        raise ValueError("Support must be in [0, 1] and lift must be non-negative")

    candidates = pairs.loc[
        pairs["prior_support"].ge(support_threshold)
        & pairs["prior_lift"].gt(lift_threshold)
    ].copy()
    candidates["ranking_confidence"] = candidates[
        ["confidence_a_to_b", "confidence_b_to_a"]
    ].max(axis=1)
    return candidates.sort_values(
        ["prior_lift", "ranking_confidence", "prior_support"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def _read_pair_counts(
    connection: duckdb.DuckDBPyConnection,
    input_dir: Path,
    prior_orders: int,
    train_orders: int,
) -> pd.DataFrame:
    minimum_pair_count = math.ceil(prior_orders * min(SUPPORT_SENSITIVITY))
    prior_path = str(input_dir / "order_products__prior.csv")
    train_path = str(input_dir / "order_products__train.csv")
    products_path = str(input_dir / "products.csv")
    aisles_path = str(input_dir / "aisles.csv")
    departments_path = str(input_dir / "departments.csv")
    connection.execute(
        """
        CREATE TEMP TABLE prior_item_counts AS
        SELECT product_id, COUNT(*)::BIGINT AS order_count
        FROM read_csv_auto(?)
        GROUP BY product_id
        """,
        [prior_path],
    )
    connection.execute(
        """
        CREATE TEMP TABLE prior_pair_counts AS
        SELECT a.product_id AS item_a,
               b.product_id AS item_b,
               COUNT(*)::BIGINT AS pair_count
        FROM read_csv_auto(?) AS a
        INNER JOIN prior_item_counts AS ca ON ca.product_id = a.product_id
        INNER JOIN read_csv_auto(?) AS b
            ON b.order_id = a.order_id AND b.product_id > a.product_id
        INNER JOIN prior_item_counts AS cb ON cb.product_id = b.product_id
        WHERE ca.order_count >= ? AND cb.order_count >= ?
        GROUP BY a.product_id, b.product_id
        HAVING COUNT(*) >= ?
        """,
        [prior_path, prior_path, minimum_pair_count, minimum_pair_count, minimum_pair_count],
    )
    connection.execute(
        """
        CREATE TEMP TABLE train_item_counts AS
        SELECT product_id, COUNT(*)::BIGINT AS order_count
        FROM read_csv_auto(?)
        GROUP BY product_id
        """,
        [train_path],
    )
    connection.execute(
        """
        CREATE TEMP TABLE train_pair_counts AS
        SELECT candidate.item_a, candidate.item_b, COUNT(*)::BIGINT AS pair_count
        FROM read_csv_auto(?) AS a
        INNER JOIN read_csv_auto(?) AS b
            ON b.order_id = a.order_id AND b.product_id > a.product_id
        INNER JOIN prior_pair_counts AS candidate
            ON candidate.item_a = a.product_id AND candidate.item_b = b.product_id
        GROUP BY candidate.item_a, candidate.item_b
        """,
        [train_path, train_path],
    )
    return connection.execute(
        """
        SELECT prior.item_a,
               prior.item_b,
               prior.pair_count AS prior_pair_orders,
               prior.pair_count::DOUBLE / ? AS prior_support,
               prior.pair_count::DOUBLE / a.order_count AS confidence_a_to_b,
               prior.pair_count::DOUBLE / b.order_count AS confidence_b_to_a,
               (prior.pair_count::DOUBLE * ?) /
                   (a.order_count * b.order_count) AS prior_lift,
               COALESCE(train_pair.pair_count, 0) AS train_pair_orders,
               COALESCE(train_pair.pair_count, 0)::DOUBLE / ? AS train_support,
               COALESCE(train_pair.pair_count, 0)::DOUBLE /
                   NULLIF(train_a.order_count, 0) AS train_confidence_a_to_b,
               COALESCE(train_pair.pair_count, 0)::DOUBLE /
                   NULLIF(train_b.order_count, 0) AS train_confidence_b_to_a,
               (COALESCE(train_pair.pair_count, 0)::DOUBLE * ?) /
                   NULLIF(train_a.order_count * train_b.order_count, 0) AS train_lift,
               product_a.product_name AS product_a_name,
               product_b.product_name AS product_b_name,
               aisle_a.aisle AS aisle_a,
               aisle_b.aisle AS aisle_b,
               department_a.department AS department_a,
               department_b.department AS department_b
        FROM prior_pair_counts AS prior
        INNER JOIN prior_item_counts AS a ON a.product_id = prior.item_a
        INNER JOIN prior_item_counts AS b ON b.product_id = prior.item_b
        LEFT JOIN train_pair_counts AS train_pair
            ON train_pair.item_a = prior.item_a AND train_pair.item_b = prior.item_b
        LEFT JOIN train_item_counts AS train_a ON train_a.product_id = prior.item_a
        LEFT JOIN train_item_counts AS train_b ON train_b.product_id = prior.item_b
        LEFT JOIN read_csv_auto(?) AS product_a ON product_a.product_id = prior.item_a
        LEFT JOIN read_csv_auto(?) AS product_b ON product_b.product_id = prior.item_b
        LEFT JOIN read_csv_auto(?) AS aisle_a ON aisle_a.aisle_id = product_a.aisle_id
        LEFT JOIN read_csv_auto(?) AS aisle_b ON aisle_b.aisle_id = product_b.aisle_id
        LEFT JOIN read_csv_auto(?) AS department_a
            ON department_a.department_id = product_a.department_id
        LEFT JOIN read_csv_auto(?) AS department_b
            ON department_b.department_id = product_b.department_id
        """,
        [
            prior_orders,
            prior_orders,
            train_orders,
            train_orders,
            products_path,
            products_path,
            aisles_path,
            aisles_path,
            departments_path,
            departments_path,
        ],
    ).df()


def _sample_fp_growth(
    input_dir: Path,
    candidates: pd.DataFrame,
    sample_size: int = FP_GROWTH_SAMPLE_SIZE,
) -> pd.DataFrame:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if len(candidates) < 3:
        raise ValueError("At least three bundle candidates are required for FP-Growth check")

    selected_pairs = candidates.head(3)[["item_a", "item_b"]].copy()
    product_ids = sorted(
        set(selected_pairs["item_a"].astype(int))
        | set(selected_pairs["item_b"].astype(int))
    )
    with tempfile.TemporaryDirectory(prefix="instacart-fpgrowth-") as temp_dir:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit='768MB'")
            connection.execute("SET threads=2")
            connection.execute("SET temp_directory=?", [temp_dir])
            connection.execute(
                """
                CREATE TEMP TABLE sampled_orders AS
                SELECT order_id
                FROM (
                    SELECT DISTINCT order_id
                    FROM read_csv_auto(?)
                )
                ORDER BY hash(order_id)
                LIMIT ?
                """,
                [str(input_dir / "order_products__prior.csv"), sample_size],
            )
            actual_sample_size = int(
                connection.execute("SELECT COUNT(*) FROM sampled_orders").fetchone()[0]
            )
            if actual_sample_size != sample_size:
                raise ValueError(
                    f"Requested {sample_size:,} sample orders but found "
                    f"{actual_sample_size:,}"
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
            sampled_rows = connection.execute(
                """
                SELECT DISTINCT sampled.order_id, items.product_id
                FROM read_csv_auto(?) AS items
                INNER JOIN sampled_orders AS sampled USING (order_id)
                WHERE items.product_id = ANY(?)
                """,
                [str(input_dir / "order_products__prior.csv"), product_ids],
            ).df()
        finally:
            connection.close()

    # Map the restricted pair-item universe to a sparse order-by-product matrix.
    if len(sample_order_ids) != sample_size:
        raise RuntimeError("The selected order sample has an unexpected row count")
    sparse_baskets = pd.DataFrame(
        {
            product_id: pd.arrays.SparseArray(
                np.zeros(sample_size, dtype=bool), fill_value=False
            )
            for product_id in product_ids
        }
    )
    sparse_positions = pd.Series(
        np.arange(sample_size, dtype=np.int64), index=sample_order_ids
    )
    for product_id, group in sampled_rows.groupby("product_id", sort=False):
        positions = sparse_positions.loc[group["order_id"].astype("int64")].to_numpy()
        values = np.zeros(sample_size, dtype=bool)
        values[positions] = True
        sparse_baskets[int(product_id)] = pd.arrays.SparseArray(values, fill_value=False)
    sparse_baskets.columns = [str(int(product_id)) for product_id in sparse_baskets.columns]

    itemsets = fpgrowth(
        sparse_baskets,
        min_support=1 / sample_size,
        use_colnames=True,
        max_len=2,
    )
    pair_support = {
        frozenset(int(product_id) for product_id in itemset): float(support)
        for itemset, support in zip(itemsets["itemsets"], itemsets["support"], strict=True)
        if len(itemset) == 2
    }

    direct_counts = sampled_rows.groupby(
        ["order_id", "product_id"], sort=False
    ).size()
    direct_counts = direct_counts.reset_index(name="row_count")
    direct_sets = direct_counts.groupby("order_id")["product_id"].agg(set)
    result_rows = []
    for item_a, item_b in selected_pairs.itertuples(index=False, name=None):
        count = sum(
            int(item_a in basket and item_b in basket)
            for basket in direct_sets
        )
        direct_support = count / sample_size
        fp_support = pair_support.get(frozenset((int(item_a), int(item_b))), 0.0)
        result_rows.append(
            {
                "item_a": int(item_a),
                "item_b": int(item_b),
                "sample_orders": sample_size,
                "direct_pair_orders": count,
                "direct_support": direct_support,
                "fpgrowth_support": fp_support,
                "absolute_difference": abs(direct_support - fp_support),
                "matches": bool(np.isclose(direct_support, fp_support)),
            }
        )
    result = pd.DataFrame(result_rows)
    if not result["matches"].all():
        raise AssertionError("FP-Growth support did not match direct sample counts")
    return result


def _source_audit(input_dir: Path) -> dict[str, Any]:
    _validate_input_files(input_dir)
    files = {}
    connection = duckdb.connect(":memory:")
    try:
        for name in REQUIRED_FILES:
            path = input_dir / name
            files[name] = {
                "rows": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM read_csv_auto(?)", [str(path)]
                    ).fetchone()[0]
                ),
                "bytes": path.stat().st_size,
                "sha256": _file_hash(path),
                "columns": [
                    row[0]
                    for row in connection.execute(
                        "DESCRIBE SELECT * FROM read_csv_auto(?)", [str(path)]
                    ).fetchall()
                ],
            }
        orders = str(input_dir / "orders.csv")
        prior = str(input_dir / "order_products__prior.csv")
        order_sets = connection.execute(
            "SELECT eval_set, COUNT(*) FROM read_csv_auto(?) GROUP BY eval_set ORDER BY eval_set",
            [orders],
        ).fetchall()
        join_counts = connection.execute(
            """
            SELECT COUNT(*) FILTER (WHERE o.order_id IS NULL),
                   COUNT(*) FILTER (WHERE p.product_id IS NULL),
                   COUNT(*) FILTER (WHERE a.aisle_id IS NULL),
                   COUNT(*) FILTER (WHERE d.department_id IS NULL),
                   COUNT(*) - COUNT(DISTINCT (op.order_id, op.product_id))
            FROM read_csv_auto(?) AS op
            LEFT JOIN read_csv_auto(?) AS o USING (order_id)
            LEFT JOIN read_csv_auto(?) AS p USING (product_id)
            LEFT JOIN read_csv_auto(?) AS a USING (aisle_id)
            LEFT JOIN read_csv_auto(?) AS d USING (department_id)
            """,
            [
                prior,
                orders,
                str(input_dir / "products.csv"),
                str(input_dir / "aisles.csv"),
                str(input_dir / "departments.csv"),
            ],
        ).fetchone()
    finally:
        connection.close()
    return {
        "source_files": files,
        "order_eval_sets": [
            {"eval_set": row[0], "orders": int(row[1])} for row in order_sets
        ],
        "prior_missing_orders": int(join_counts[0]),
        "prior_missing_products": int(join_counts[1]),
        "prior_missing_aisles": int(join_counts[2]),
        "prior_missing_departments": int(join_counts[3]),
        "prior_duplicate_order_product_rows": int(join_counts[4]),
    }


def run_bundle_analysis(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    lift_threshold: float = DEFAULT_LIFT_THRESHOLD,
    sample_size: int = FP_GROWTH_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Generate traceable bundle candidates, source audit, and FP-Growth checks."""
    if not 0 <= support_threshold <= 1 or lift_threshold < 0:
        raise ValueError("Support must be in [0, 1] and lift must be non-negative")
    audit = _source_audit(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="instacart-pairs-") as temp_dir:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit='768MB'")
            connection.execute("SET threads=2")
            connection.execute("SET temp_directory=?", [temp_dir])
            eval_sets = {
                entry["eval_set"]: entry["orders"]
                for entry in audit["order_eval_sets"]
            }
            prior_orders = eval_sets["prior"]
            train_orders = eval_sets["train"]
            pair_metrics = _read_pair_counts(
                connection, input_dir, prior_orders, train_orders
            )
        finally:
            connection.close()

    pair_metrics["m5_category"] = [
        map_m5_category(dept_a, dept_b)
        for dept_a, dept_b in zip(
            pair_metrics["department_a"],
            pair_metrics["department_b"],
            strict=True,
        )
    ]
    sensitivity_rows = []
    for threshold in SUPPORT_SENSITIVITY:
        threshold_pairs = pair_metrics.loc[
            pair_metrics["prior_support"].ge(threshold)
            & pair_metrics["prior_lift"].gt(lift_threshold)
        ]
        valid_correlation = threshold_pairs[
            ["prior_lift", "train_lift"]
        ].dropna()
        correlation = (
            float(valid_correlation["prior_lift"].corr(valid_correlation["train_lift"]))
            if len(valid_correlation) > 1
            else float("nan")
        )
        sensitivity_rows.append(
            {
                "minimum_prior_support": threshold,
                "minimum_prior_pair_orders": math.ceil(prior_orders * threshold),
                "pairs_passing_prior_lift": int(len(threshold_pairs)),
                "share_train_lift_above_threshold": (
                    float(threshold_pairs["train_lift"].gt(lift_threshold).mean())
                    if len(threshold_pairs)
                    else float("nan")
                ),
                "prior_train_lift_correlation": correlation,
                "correlation_pair_count": int(len(valid_correlation)),
                "pairs_mapped_to_m5_category": int(
                    threshold_pairs["m5_category"].ne("").sum()
                ),
                "m5_category_mapping_share": (
                    float(threshold_pairs["m5_category"].ne("").mean())
                    if len(threshold_pairs)
                    else float("nan")
                ),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    candidates = rank_bundle_candidates(
        pair_metrics, support_threshold, lift_threshold
    )
    candidates["validation_status"] = VALIDATION_STATUS
    output_columns = [
        "item_a",
        "product_a_name",
        "aisle_a",
        "department_a",
        "item_b",
        "product_b_name",
        "aisle_b",
        "department_b",
        "prior_support",
        "confidence_a_to_b",
        "confidence_b_to_a",
        "prior_lift",
        "train_lift",
        "m5_category",
        "ranking_confidence",
        "prior_pair_orders",
        "train_pair_orders",
        "validation_status",
    ]
    candidates = candidates[output_columns]
    pair_crosscheck = _sample_fp_growth(
        input_dir,
        rank_bundle_candidates(pair_metrics, support_threshold, lift_threshold),
        sample_size,
    )
    candidates.to_csv(output_dir / "bundle_candidates.csv", index=False)
    sensitivity.to_csv(output_dir / "bundle_support_sensitivity.csv", index=False)
    pair_crosscheck.to_csv(output_dir / "fpgrowth_sample_crosscheck.csv", index=False)
    audit_path = output_dir / "instacart_data_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(
        output_dir,
        candidates,
        sensitivity,
        pair_crosscheck,
        support_threshold,
        lift_threshold,
        sample_size,
        audit,
    )
    return candidates


def _write_summary(
    output_dir: Path,
    candidates: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fp_check: pd.DataFrame,
    support_threshold: float,
    lift_threshold: float,
    sample_size: int,
    audit: dict[str, Any],
) -> None:
    main = sensitivity.loc[
        np.isclose(sensitivity["minimum_prior_support"], support_threshold)
    ].iloc[0]
    sensitivity_lines = "\n".join(
        f"| {row.minimum_prior_support:.2%} | "
        f"{int(row.pairs_passing_prior_lift):,} | "
        f"{row.share_train_lift_above_threshold:.1%} | "
        f"{row.prior_train_lift_correlation:.3f} "
        f"(n={int(row.correlation_pair_count):,}) | "
        f"{int(row.pairs_mapped_to_m5_category):,} |"
        for row in sensitivity.itertuples(index=False)
    )
    fp_lines = "\n".join(
        f"| {row.item_a} + {row.item_b} | {row.sample_orders:,} | "
        f"{row.direct_support:.5%} | {row.fpgrowth_support:.5%} | "
        f"{'yes' if row.matches else 'no'} |"
        for row in fp_check.itertuples(index=False)
    )
    file_lines = "\n".join(
        f"- `{name}` — {details['rows']:,} rows; SHA-256 `{details['sha256']}`"
        for name, details in audit["source_files"].items()
    )
    mapped_pair_count = int(candidates["m5_category"].ne("").sum())
    unmapped_pair_count = len(candidates) - mapped_pair_count
    department_pair_counts = (
        pd.concat(
            [
                candidates[["item_a", "item_b", "department_a"]].rename(
                    columns={"department_a": "department"}
                ),
                candidates[["item_a", "item_b", "department_b"]].rename(
                    columns={"department_b": "department"}
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates(["item_a", "item_b", "department"])
        .groupby("department", sort=True)
        .size()
    )
    department_lines = "\n".join(
        f"| {department} | {int(pair_count):,} |"
        for department, pair_count in department_pair_counts.items()
    )
    category_pair_counts = (
        candidates["m5_category"]
        .replace("", "Unmapped")
        .value_counts()
        .sort_index()
    )
    category_lines = "\n".join(
        f"| {category} | {int(pair_count):,} |"
        for category, pair_count in category_pair_counts.items()
    )
    top_pair_lines = "\n".join(
        f"| {row.product_a_name} ({row.department_a}) + "
        f"{row.product_b_name} ({row.department_b}) | "
        f"{row.prior_lift:.2f} | {row.prior_support:.3%} |"
        for row in candidates.head(10).itertuples(index=False)
    )
    report = f"""# Module 5 — bundle candidates

## Provenance and method

Source files in `data/raw/instacart/`:
{file_lines}

Pairs are unordered product pairs counted once per order. Discovery support,
directional confidence, and lift are computed from prior orders. The working
rule keeps pairs with prior support at least {support_threshold:.2%} and prior
lift strictly above {lift_threshold:.2f}. Results rank by prior lift, maximum
directional confidence, then support. The two directional confidence values
remain in the output. Every candidate is labelled
“{VALIDATION_STATUS}”.

## Threshold stability and category mapping

“Train lift retained” is the share of pairs passing each prior-support and
prior-lift rule whose train lift is also strictly above {lift_threshold:.2f}.
Lift correlation is Pearson correlation among pairs with defined prior and
train lift. Train orders overlap the same users as prior history, so this is
not independent-user validation.

| Minimum prior support | Pairs passing prior lift | Share train lift > {lift_threshold:.1f} | Prior/train lift correlation | Pairs with M5 category |
|---:|---:|---:|---:|---:|
{sensitivity_lines}

At the working {support_threshold:.2%} support threshold,
{int(main.pairs_mapped_to_m5_category):,} of {len(candidates):,} output pairs
({main.pairs_mapped_to_m5_category / len(candidates):.1%}) map to one M5
category. A pair maps only when both product departments map to the same
M5 category; unmapped, cross-category, and non-food products have a blank
`m5_category`.

{mapped_pair_count:,} pairs mapped, {unmapped_pair_count:,} unmapped. The mapping is an assumption, not data.

### Candidate pairs by department

Each candidate is counted once per department it contains, even when both
products in the pair are from the same department.

| Department | Candidate pairs |
|---|---:|
{department_lines}

### Candidate pairs by mapped M5 category

| M5 category | Candidate pairs |
|---|---:|
{category_lines}

### Top 10 rules by prior lift

These rules are ranked by lift, then maximum directional confidence, then
support. The highest-lift pair is a Greek-yogurt flavor pair; other leading
examples include sparkling waters and common produce combinations. Produce
dominates the candidate department distribution.

| Product pair (department) | Prior lift | Prior support |
|---|---:|---:|
{top_pair_lines}

## FP-Growth cross-check

The top three ranked candidate pairs were re-counted with mlxtend FP-Growth
on a deterministic {sample_size:,}-prior-order hash sample. The sparse basket
matrix is restricted to those six pair products to keep memory bounded;
support still uses all sampled orders as the denominator.

| Pair IDs | Sample orders | Direct pair support | FP-Growth support | Match |
|---|---:|---:|---:|---|
{fp_lines}

## Interpretation and limitations

The resulting output is
`data/processed/module5/bundle_candidates.csv`. “Bought together” means
observed basket co-occurrence; it does **not** mean the products respond to
each other’s price changes. Association is not evidence of cross-price
elasticity, complementarity, causal bundle uplift, or incremental profit.
Validate cross-price response with a designed experiment or a credible causal
method before making pricing or promotion decisions.

The department-to-M5 mapping is an explicit taxonomy assumption, not a
crosswalk supplied with Instacart or M5. See `assumptions.md`.
"""
    (output_dir / "bundle_candidate_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-support",
        type=float,
        default=DEFAULT_SUPPORT_THRESHOLD,
        help="Minimum prior-order pair support (default: 0.25%%)",
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=DEFAULT_LIFT_THRESHOLD,
        help="Strict prior-lift cutoff (default: >1.5)",
    )
    parser.add_argument(
        "--fpgrowth-sample-size",
        type=int,
        default=FP_GROWTH_SAMPLE_SIZE,
        help="Prior-order sample size for the mlxtend FP-Growth cross-check",
    )
    args = parser.parse_args()
    candidates = run_bundle_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        support_threshold=args.min_support,
        lift_threshold=args.min_lift,
        sample_size=args.fpgrowth_sample_size,
    )
    print(f"Bundle candidates: {len(candidates):,}")
    print(f"Output: {args.output_dir / 'bundle_candidates.csv'}")
    print(f"Summary: {args.output_dir / 'bundle_candidate_summary.md'}")


if __name__ == "__main__":
    main()
