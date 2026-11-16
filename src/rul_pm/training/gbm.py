import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from rul_pm.data.contract import build_contract
from rul_pm.data.labels import RUL_CAP
from rul_pm.training.ablation import META

# Pre-registered in D12. Do not change after seeing results.
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=0,
    verbose=-1,
)


def make_lgbm(base_cols: list[str], window: int | None = None, params: dict | None = None):
    """LightGBM on contract features, optionally plus trailing rolling means."""
    cfg = {**LGBM_PARAMS, **(params or {})}

    def predict(tr: pd.DataFrame, va: pd.DataFrame) -> np.ndarray:
        feats = list(build_contract(tr[base_cols])["features"])
        if window is not None:
            feats = feats + [f"{c}_roll{window}" for c in feats if c not in META]
        model = LGBMRegressor(**cfg)
        model.fit(tr[feats], tr["rul"])
        return np.clip(model.predict(va[feats]), 0, RUL_CAP)

    return predict