import numpy as np
import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import COLUMNS
from rul_pm.training.baselines import evaluate, predict_age


def _fleet():
    rng = np.random.default_rng(0)
    parts = []
    for u in range(1, 11):
        n = 150 + 5 * u
        d = pd.DataFrame(rng.normal(size=(n, len(COLUMNS))), columns=COLUMNS)
        d["unit"], d["cycle"] = u, np.arange(1, n + 1)
        parts.append(d)
    return add_rul(pd.concat(parts, ignore_index=True))


def test_age_baseline_hand_values():
    out = predict_age(_fleet(), pd.DataFrame({"cycle": [1, 100, 190]}))
    assert out.tolist() == [125.0, 77.5, 0.0]


def test_evaluate_shape_and_finite():
    res = evaluate(_fleet(), seeds=range(2))
    assert len(res) == 3 * 2 * 2  # models x seeds x truths
    assert np.isfinite(res[["nasa", "mae", "rmse"]].to_numpy()).all()
