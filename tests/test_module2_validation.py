import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("doubleml")
pytest.importorskip("xgboost")

from retail_engine.module2_validation import run_c1_sensitivity


def _write_tutorial_subset(path, rows=800, elasticity=-1.5, seed=0):
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        rng.normal(size=(rows, 5)), columns=["f1", "f2", "f3", "f4", "f5"]
    )
    d = 0.8 * features["f1"] + rng.normal(0, 0.5, rows)
    y = elasticity * d + 1.0 * features["f2"] + rng.normal(0, 0.5, rows)
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "Quantity": 1.0,
                    "revenue": 1.0,
                    "UnitPrice": 1.0,
                    "dLnQ": y,
                    "dLnP": d,
                }
            ),
            features,
        ],
        axis=1,
    )
    frame.to_csv(path)


def test_c1_sensitivity_reports_bounds_robustness_and_benchmarks(tmp_path):
    data_path = tmp_path / "elasticity_subset.csv"
    _write_tutorial_subset(data_path)

    audit = run_c1_sensitivity(
        data_path, tmp_path / "out", n_folds=2, n_estimators=30, top_features=3
    )

    assert audit["elasticity"] == pytest.approx(-1.5, abs=0.25)
    lower, upper = audit["theta_bounds"]
    assert lower <= audit["elasticity"] <= upper
    assert audit["ci_bounds"][0] <= audit["ci_lower"]
    assert 0.0 < audit["robustness_value"] < 1.0
    benchmarks = pd.read_csv(tmp_path / "out" / "c1_doubleml_sensitivity_benchmarks.csv")
    assert len(benchmarks) == 6
    assert set(benchmarks["top_predictor_of"]) == {"outcome", "treatment"}
    saved = json.loads((tmp_path / "out" / "c1_doubleml_sensitivity_audit.json").read_text())
    assert saved["robustness_value"] == audit["robustness_value"]
