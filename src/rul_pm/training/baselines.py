import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rul_pm.data.contract import build_contract
from rul_pm.data.labels import RUL_CAP
from rul_pm.evaluation.metrics import error_summary
from rul_pm.evaluation.validation import group_folds, sample_cuts

SEEDS = range(5)


def predict_mean(tr, va):
    return np.full(len(va), tr["rul"].mean())


def predict_age(tr, va):
    mean_life = tr.groupby("unit")["cycle"].max().mean()
    return np.clip(mean_life - va["cycle"].to_numpy(), 0, RUL_CAP)


def predict_ridge(tr, va):
    feats = build_contract(tr)["features"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(tr[feats], tr["rul"])
    return np.clip(model.predict(va[feats]), 0, RUL_CAP)


MODELS = {"mean": predict_mean, "age": predict_age, "ridge": predict_ridge}


def evaluate(df, models=MODELS, n_splits=5, seeds=SEEDS) -> pd.DataFrame:
    folds = group_folds(df, n_splits)
    rows = []
    for seed in seeds:
        parts = {name: [] for name in models}
        for tr_idx, va_idx in folds:
            tr, va = df.iloc[tr_idx], df.iloc[va_idx]
            cuts = sample_cuts(va, seed)
            for name, fn in models.items():
                parts[name].append(cuts.assign(pred=fn(tr, cuts)))
        for name, ps in parts.items():
            oof = pd.concat(ps)
            for truth in ("rul", "rul_raw"):
                s = error_summary(oof[truth], oof["pred"])
                rows.append({"model": name, "seed": seed, "truth": truth, **s})
    return pd.DataFrame(rows)
