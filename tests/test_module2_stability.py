import pandas as pd

from retail_engine.module2_stability import same_item_stability


def _row(item, store, elasticity, lower, upper, status="estimated"):
    return {
        "item_id": item,
        "store_id": store,
        "elasticity": elasticity,
        "ci_lower": lower,
        "ci_upper": upper,
        "fit_status": status,
    }


def test_stability_summarises_items_seen_in_several_stores():
    results = pd.DataFrame(
        [
            _row("A", "CA_1", -2.0, -2.5, -1.5),
            _row("A", "TX_1", -3.0, -3.5, -2.5),  # both price-sensitive, overlapping
            _row("B", "CA_1", -2.0, -2.5, -1.5),
            _row("B", "TX_1", 1.0, 0.5, 1.5),  # opposite sign, disjoint intervals
            _row("C", "CA_1", -0.5, -1.0, 0.0),  # only one store
            _row("D", "CA_1", -1.0, -2.0, 0.0),
            _row("D", "TX_1", -1.0, -2.0, 0.0, status="not_estimable"),
        ]
    )

    table = same_item_stability(results).set_index("item_id")

    assert list(table.index) == ["A", "B"]  # C has one store; D has one fitted store
    a = table.loc["A"]
    assert a["store_count"] == 2
    assert a["elasticity_spread"] == 1.0
    assert a["sign_agreement_share"] == 1.0
    assert a["modal_demand_type"] == "Price-sensitive"
    assert a["demand_type_agreement_share"] == 1.0
    assert bool(a["common_interval_exists"])

    b = table.loc["B"]
    assert b["sign_agreement_share"] == 0.5
    assert not bool(b["common_interval_exists"])


def test_stability_uses_only_fitted_series_for_counts():
    results = pd.DataFrame(
        [
            _row("E", "CA_1", -2.0, -2.5, -1.5),
            _row("E", "TX_1", float("nan"), float("nan"), float("nan"), status="not_estimable"),
        ]
    )

    assert same_item_stability(results).empty
