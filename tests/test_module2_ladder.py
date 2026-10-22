import numpy as np
import pandas as pd
import pytest

from retail_engine.module2_ladder import SPECIFICATIONS, control_ladder


def _series(item_id, slope, seed):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2015-01-03", periods=120, freq="7D")
    log_price = rng.normal(0, 0.15, len(weeks))
    units = np.exp(2.0 + slope * log_price + rng.normal(0, 0.05, len(weeks))) - 1
    return pd.DataFrame(
        {
            "item_id": item_id,
            "store_id": "CA_1",
            "week_start": weeks,
            "units_sold": units,
            "price": np.exp(log_price),
            "snap_share": rng.random(len(weeks)),
            "event_week": rng.integers(0, 2, len(weeks)).astype(float),
            "promo_week": rng.integers(0, 2, len(weeks)).astype(float),
        }
    )


def test_ladder_has_one_row_per_series_and_all_specifications():
    weekly = pd.concat([_series("A", -1.5, 1), _series("B", -0.5, 2)])

    ladder = control_ladder(weekly)

    assert list(ladder["item_id"]) == ["A", "B"]
    assert list(ladder.columns[2:]) == list(SPECIFICATIONS)
    assert ladder[list(SPECIFICATIONS)].notna().all().all()


def test_ladder_recovers_known_elasticity_when_controls_are_irrelevant():
    ladder = control_ladder(_series("A", -1.5, 3)).iloc[0]

    assert ladder["no_controls"] == pytest.approx(-1.5, abs=0.2)
    assert ladder["seasonality_snap_event"] == pytest.approx(-1.5, abs=0.2)
