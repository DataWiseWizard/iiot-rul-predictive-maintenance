import numpy as np
import pandas as pd

from rul_pm.data.contract import CANDIDATES
from rul_pm.data.labels import RUL_CAP
from rul_pm.features.features import add_rolling_means
from rul_pm.training.gbm import make_lgbm

BASE = ["unit", "cycle", *CANDIDATES, "rul", "rul_raw"]
SMALL = {"n_estimators": 20, "min_child_samples": 2}


def _split(window=3):
    rng = np.random.default_rng(0)
    rows = []
    for unit, life in [(1, 20), (2, 25), (3, 30)]:
        vals = rng.normal(size=(life, len(CANDIDATES)))
        for i in range(life):
            c = i + 1
            row = {"unit": unit, "cycle": c,
                   "rul": float(min(125, life - c)), "rul_raw": float(life - c)}
            row.update(zip(CANDIDATES, vals[i], strict=True))
            rows.append(row)
    rolled = add_rolling_means(pd.DataFrame(rows), list(CANDIDATES), window=window)
    return rolled[rolled.unit != 3], rolled[rolled.unit == 3]


def test_predictions_are_one_per_row_finite_and_clipped():
    tr, va = _split()
    pred = make_lgbm(BASE, params=SMALL)(tr, va)
    assert len(pred) == len(va)
    assert np.isfinite(pred).all()
    assert pred.min() >= 0 and pred.max() <= RUL_CAP


def test_rolling_variant_uses_different_inputs():
    tr, va = _split()
    raw = make_lgbm(BASE, params=SMALL)(tr, va)
    roll = make_lgbm(BASE, window=3, params=SMALL)(tr, va)
    assert not np.allclose(raw, roll)