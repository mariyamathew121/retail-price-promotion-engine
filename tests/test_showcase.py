import pytest

from retail_engine.showcase import simulate_change


def test_unit_elasticity_gives_equal_units_and_revenue_loss_on_a_price_rise():
    result = simulate_change(100.0, 2.0, elasticity=-1.0, price_change=0.10)

    assert result["units_change_pct"] == pytest.approx(-9.0909, rel=1e-3)
    assert result["revenue_change_pct"] == pytest.approx(0.0, abs=1e-9)
    assert result["price"] == pytest.approx(2.2)


def test_elastic_product_loses_revenue_on_a_price_rise():
    result = simulate_change(100.0, 2.0, elasticity=-2.0, price_change=0.10)

    assert result["revenue_change_pct"] < 0


def test_no_price_change_means_no_change():
    result = simulate_change(50.0, 3.0, elasticity=-1.5, price_change=0.0)

    assert result["units"] == pytest.approx(50.0)
    assert result["revenue_change_pct"] == pytest.approx(0.0)
