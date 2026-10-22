import pandas as pd
import pytest

from retail_engine.module5 import (
    VALIDATION_STATUS,
    _sample_fp_growth,
    map_m5_category,
    pair_statistics,
    rank_bundle_candidates,
)


def test_pair_statistics_match_hand_calculated_example():
    metrics = pair_statistics(
        pair_orders=20,
        item_a_orders=100,
        item_b_orders=80,
        total_orders=1000,
    )

    assert metrics["support"] == pytest.approx(0.02)
    assert metrics["confidence_a_to_b"] == pytest.approx(0.20)
    assert metrics["confidence_b_to_a"] == pytest.approx(0.25)
    assert metrics["prior_lift"] == pytest.approx(2.5)


def test_candidate_filter_uses_strict_lift_and_ranks_by_lift_confidence_support():
    pairs = pd.DataFrame(
        [
            {
                "item_a": 1,
                "item_b": 2,
                "prior_support": 0.01,
                "confidence_a_to_b": 0.40,
                "confidence_b_to_a": 0.50,
                "prior_lift": 2.0,
            },
            {
                "item_a": 3,
                "item_b": 4,
                "prior_support": 0.02,
                "confidence_a_to_b": 0.20,
                "confidence_b_to_a": 0.30,
                "prior_lift": 2.0,
            },
            {
                "item_a": 5,
                "item_b": 6,
                "prior_support": 0.03,
                "confidence_a_to_b": 0.90,
                "confidence_b_to_a": 0.90,
                "prior_lift": 1.5,
            },
            {
                "item_a": 7,
                "item_b": 8,
                "prior_support": 0.002,
                "confidence_a_to_b": 0.90,
                "confidence_b_to_a": 0.90,
                "prior_lift": 3.0,
            },
        ]
    )

    candidates = rank_bundle_candidates(pairs)

    assert list(zip(candidates["item_a"], candidates["item_b"])) == [(1, 2), (3, 4)]
    assert candidates.loc[0, "ranking_confidence"] == pytest.approx(0.50)


@pytest.mark.parametrize(
    ("department_a", "department_b", "expected"),
    [
        ("household", "household", "HOUSEHOLD"),
        ("produce", "dairy eggs", "FOODS"),
        ("pets", "produce", ""),
        ("babies", "personal care", ""),
        ("bakery", "household", ""),
    ],
)
def test_instacart_departments_map_only_same_m5_category(
    department_a: str,
    department_b: str,
    expected: str,
):
    assert map_m5_category(department_a, department_b) == expected


def test_fp_growth_crosscheck_matches_three_pair_supports(tmp_path):
    prior = tmp_path / "order_products__prior.csv"
    prior.write_text(
        "order_id,product_id,add_to_cart_order,reordered\n"
        "1,1,1,0\n1,2,2,0\n"
        "2,1,1,0\n2,2,2,0\n"
        "3,3,1,0\n3,4,2,0\n"
        "4,5,1,0\n4,6,2,0\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        {"item_a": [1, 3, 5], "item_b": [2, 4, 6]}
    )

    result = _sample_fp_growth(tmp_path, candidates, sample_size=4)

    assert result["matches"].all()
    assert result["direct_support"].tolist() == pytest.approx([0.5, 0.25, 0.25])
    assert result["fpgrowth_support"].tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_validation_status_value_is_fixed():
    assert VALIDATION_STATUS == "Needs cross-price validation"


def _write_instacart_fixture(tmp_path):
    baskets = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 3], [3, 4], [3, 4], [5], [5], [5]]
    lines = ["order_id,product_id,add_to_cart_order,reordered"]
    for order, basket in enumerate(baskets, start=1):
        for position, product in enumerate(basket, start=1):
            lines.append(f"{order},{product},{position},0")
    (tmp_path / "order_products__prior.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (tmp_path / "products.csv").write_text(
        "product_id,product_name,aisle_id,department_id\n"
        "1,Pasta,1,1\n2,Pasta Sauce,2,1\n3,Soap,3,2\n4,Sponge,4,2\n5,Milk,5,3\n",
        encoding="utf-8",
    )
    (tmp_path / "departments.csv").write_text(
        "department_id,department\n1,dry goods pasta\n2,household\n3,dairy eggs\n",
        encoding="utf-8",
    )


def test_fpgrowth_rules_are_directional_ranked_and_flagged(tmp_path):
    from retail_engine.module5 import RULE_COLUMNS, mine_fpgrowth_rules

    _write_instacart_fixture(tmp_path)

    rules = mine_fpgrowth_rules(
        tmp_path,
        sample_size=10,
        min_support=0.2,
        min_lift=1.5,
        candidate_pairs={(1, 2)},
    )

    assert list(rules.columns) == RULE_COLUMNS
    assert (rules["lift"] > 1.5).all()
    assert rules["validation_status"].eq(VALIDATION_STATUS).all()
    assert rules["rank"].tolist() == list(range(1, len(rules) + 1))
    ranking = list(zip(rules["lift"], rules["confidence"], rules["support"]))
    assert ranking == sorted(ranking, reverse=True)

    forward = rules.loc[rules["rule"].eq("IF Pasta THEN Pasta Sauce")].iloc[0]
    reverse = rules.loc[rules["rule"].eq("IF Pasta Sauce THEN Pasta")].iloc[0]
    # 4 of 10 orders hold both; Pasta is in 5 orders and Pasta Sauce in 4.
    assert forward["support"] == pytest.approx(0.4)
    assert forward["confidence"] == pytest.approx(4 / 5)
    assert reverse["confidence"] == pytest.approx(1.0)
    assert forward["lift"] == pytest.approx(0.4 / (0.5 * 0.4))
    assert forward["m5_category"] == "FOODS"
    assert bool(forward["in_full_data_candidates"])

    household = rules.loc[rules["rule"].eq("IF Soap THEN Sponge")].iloc[0]
    assert household["m5_category"] == "HOUSEHOLD"
    assert not bool(household["in_full_data_candidates"])
