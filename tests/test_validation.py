import math

import pandas as pd
import pytest

from rul_pm.data.labels import add_rul
from rul_pm.evaluation.metrics import error_summary, nasa_score
from rul_pm.evaluation.validation import group_folds, sample_cuts


def _fleet(n_units=20, life=200):
    rows = [(u, c) for u in range(1, n_units + 1) for c in range(1, life + 1)]
    return add_rul(pd.DataFrame(rows, columns=["unit", "cycle"]))


def test_nasa_score_hand_values_and_asymmetry():
    assert nasa_score([100], [100]) == 0
    assert nasa_score([100], [110]) == pytest.approx(math.exp(1.0) - 1)
    assert nasa_score([100], [87]) == pytest.approx(math.exp(1.0) - 1)
    assert nasa_score([100], [110]) > nasa_score([100], [90])  # late costs more


def test_error_summary_late_stats():
    s = error_summary([100, 100, 100], [110, 90, 120])
    assert s["late_fraction"] == pytest.approx(2 / 3)
    assert s["mean_late_error"] == pytest.approx(15.0)


def test_group_folds_never_split_an_engine():
    df = _fleet()
    folds = group_folds(df, 5)
    for tr, va in folds:
        assert not set(df.iloc[tr]["unit"]) & set(df.iloc[va]["unit"])
    assert sorted(i for _, va in folds for i in va) == list(range(len(df)))


def test_sample_cuts_one_row_per_engine_in_range_and_reproducible():
    df = _fleet()
    a, b = sample_cuts(df, seed=0), sample_cuts(df, seed=0)
    assert len(a) == df["unit"].nunique() and a["unit"].is_unique
    assert a["rul_raw"].between(10, 150).all()
    assert a["rul_raw"].tolist() == b["rul_raw"].tolist()
