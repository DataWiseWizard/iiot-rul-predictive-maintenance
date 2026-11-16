import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rul_pm.data.contract import build_contract
from rul_pm.data.labels import RUL_CAP

META = ("unit", "cycle", "rul", "rul_raw")


def make_ridge(base_cols: list[str], window: int | None = None):
    """Ridge on contract features, optionally plus trailing rolling means of those features.

    The rolling columns must already exist in the frame passed in (computed on
    full engines before splitting, with add_rolling_means).
    """

    def predict(tr: pd.DataFrame, va: pd.DataFrame) -> np.ndarray:
        feats = [c for c in build_contract(tr[base_cols])["features"]]
        if window is not None:
            feats = feats + [f"{c}_roll{window}" for c in feats if c not in META]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(tr[feats], tr["rul"])
        return np.clip(model.predict(va[feats]), 0, RUL_CAP)

    return predict