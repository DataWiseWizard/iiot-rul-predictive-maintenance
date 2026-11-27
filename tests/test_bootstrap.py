import numpy as np
import pandas as pd
import pytest

from rul_pm.data.labels import RUL_CAP
from rul_pm.evaluation.bootstrap import oof_predictions, paired_bootstrap
from rul_pm.evaluation.metrics import error_summary
from rul_pm.training.baselines import MODELS, evaluate


def _frame():
    rows = []
    for unit, life in [(1, 30), (2, 35), (3, 40), (4, 45), (5, 50), (6, 55)]:
        for c in range(1, life + 1):
            rows.append({"unit": unit, "cycle": c,
                         "rul": float(min(RUL_CAP, life - c)),
                         "rul_raw": float(life - c)})
    return pd.DataFrame(rows)


def test_oof_matches_evaluate_per_seed():
    df = _frame()
    models = {"mean": MODELS["mean"], "age": MODELS["age"]}
    seeds = range(2)
    oof = oof_predictions(df, models, seeds=seeds)
    res = evaluate(df, models=models, seeds=seeds)
    for m in models:
        for s in seeds:
            t = oof[(oof["model"] == m) & (oof["seed"] == s)]
            got = error_summary(t["rul"], t["pred"])["nasa"]
            want = res[(res["model"] == m) & (res["seed"] == s)
                       & (res["truth"] == "rul")]["nasa"].item()
            assert np.isclose(got, want)


def test_identical_models_give_zero_difference():
    oof = oof_predictions(_frame(), {"mean": MODELS["mean"], "age": MODELS["age"]},
                          seeds=range(2))
    r = paired_bootstrap(oof, "age", "age", n_boot=200)
    assert abs(r["point"]) < 1e-12
    assert abs(r["ci_low"]) < 1e-12 and abs(r["ci_high"]) < 1e-12


def test_callable_metric_matches_named_metric():
    oof = oof_predictions(_frame(), {"mean": MODELS["mean"], "age": MODELS["age"]},
                          seeds=range(2))
    named = paired_bootstrap(oof, "age", "mean", metric="nasa", n_boot=50)
    custom = paired_bootstrap(oof, "age", "mean",
                              metric=lambda y, p: error_summary(y, p)["nasa"], n_boot=50)
    assert custom["point"] == pytest.approx(named["point"])