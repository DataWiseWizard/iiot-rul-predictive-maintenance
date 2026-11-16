import numpy as np
import pandas as pd
from rul_pm.training.ablation import make_ridge
from rul_pm.data.contract import CANDIDATES
from rul_pm.features.features import add_rolling_means


WINDOW = 3
BASE = ["unit", "cycle", *CANDIDATES, "rul", "rul_raw"]


def _frame():
    rng = np.random.default_rng(0)
    rows = []
    for unit, life in [(1, 20), (2, 25), (3, 30)]:
        vals = rng.normal(size=(life, len(CANDIDATES)))
        for i in range(life):
            cycle = i + 1
            row = {
                "unit": unit,
                "cycle": cycle,
                "rul": float(min(125, life - cycle)),
                "rul_raw": float(life - cycle),
            }
            row.update(zip(CANDIDATES, vals[i], strict=True))
            rows.append(row)
    return pd.DataFrame(rows)


def _split():
    rolled = add_rolling_means(_frame(), list(CANDIDATES), window=WINDOW)
    tr = rolled[rolled.unit != 3]
    va = rolled[rolled.unit == 3]
    return tr, va


def test_rolling_model_returns_one_finite_prediction_per_row():
    tr, va = _split()
    pred = make_ridge(BASE, window=WINDOW)(tr, va)
    assert len(pred) == len(va)
    assert np.isfinite(pred).all()


def test_rolling_flag_changes_the_model_inputs():
    tr, va = _split()
    raw = make_ridge(BASE)(tr, va)
    roll = make_ridge(BASE, window=WINDOW)(tr, va)
    assert not np.allclose(raw, roll)