"""Shortlist for D14 (FD001). Grids are pre-registered; do not change after the run.

Windowed models (MLP, 1D CNN, GRU) are not here yet: they need a data-contract note.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rul_pm.data.contract import build_contract
from rul_pm.data.labels import RUL_CAP
from rul_pm.data.loader import SENSOR_COLS, SETTING_COLS
from rul_pm.training.baselines import predict_ridge
from rul_pm.training.gbm import make_lgbm
from rul_pm.training.tuning import tuned

QUANTILE_ALPHA = 0.35
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
ENET_GRID = ((0.01, 0.5), (0.1, 0.5), (0.1, 0.9))          # (alpha, l1_ratio)
ET_GRID = ((5, 1.0), (20, 1.0), (20, 0.5))                  # (min_samples_leaf, max_features)
LGBM_GRID = ((31, 20), (15, 100), (7, 400))                 # (num_leaves, min_child_samples)


def make_ridge(alpha):
    def predict(tr, va):
        feats = build_contract(tr)["features"]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(tr[feats], tr["rul"])
        return np.clip(model.predict(va[feats]), 0, RUL_CAP)
    return predict


def make_enet(alpha, l1_ratio):
    def predict(tr, va):
        feats = build_contract(tr)["features"]
        model = make_pipeline(
            StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
        model.fit(tr[feats], tr["rul"])
        return np.clip(model.predict(va[feats]), 0, RUL_CAP)
    return predict


def make_et(min_samples_leaf, max_features):
    def predict(tr, va):
        feats = build_contract(tr)["features"]
        model = ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=0, n_jobs=-1)
        model.fit(tr[feats], tr["rul"])
        return np.clip(model.predict(va[feats]), 0, RUL_CAP)
    return predict


def make_lgbm_q(num_leaves, min_child_samples):
    return make_lgbm(
        base_cols=SETTING_COLS + SENSOR_COLS,
        params={"objective": "quantile", "alpha": QUANTILE_ALPHA,
                "num_leaves": num_leaves, "min_child_samples": min_child_samples},
    )


_ridge_t = tuned({f"ridge_a{a:g}": make_ridge(a) for a in RIDGE_ALPHAS})
_enet_t = tuned({f"enet_a{a:g}_l{l:g}": make_enet(a, l) for a, l in ENET_GRID})
_et_t = tuned({f"et_leaf{m}_mf{f:g}": make_et(m, f) for m, f in ET_GRID})
_lgbm_t = tuned({f"lgbmq_nl{n}_mc{c}": make_lgbm_q(n, c) for n, c in LGBM_GRID})


def predict_blend(tr, va):
    return np.clip(0.5 * _ridge_t(tr, va) + 0.5 * _et_t(tr, va), 0, RUL_CAP)


TUNED = {
    "ridge_tuned": _ridge_t,
    "elastic_net_tuned": _enet_t,
    "extra_trees_tuned": _et_t,
    "lgbm_quantile_tuned": _lgbm_t,
}

SHORTLIST = {
    "ridge": predict_ridge,  # untuned continuity row (matches D12 numbers)
    **TUNED,
    "blend_tuned": predict_blend,
}