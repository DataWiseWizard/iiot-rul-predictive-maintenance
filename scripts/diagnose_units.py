import numpy as np
import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions
from rul_pm.training.ablation import make_ridge
from rul_pm.training.gbm import make_lgbm

LATE_DIV, EARLY_DIV = 10.0, 13.0  # D4
UNITS = [90, 61, 65, 57]


def row_nasa(truth, pred):
    d = pred - truth
    return np.where(d >= 0, np.exp(d / LATE_DIV) - 1, np.exp(-d / EARLY_DIV) - 1)


df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
models = {"ridge_raw": make_ridge(base_cols), "lgbm_raw": make_lgbm(base_cols)}
oof = oof_predictions(df, models)

r = oof[oof.model == "ridge_raw"].set_index(["seed", "unit", "cycle"])
l = oof[oof.model == "lgbm_raw"].set_index(["seed", "unit", "cycle"]).loc[r.index]

out = pd.DataFrame({
    "rul": r["rul"],
    "ridge_pred": r["pred"],
    "lgbm_pred": l["pred"],
    "ridge_nasa": row_nasa(r["rul"].to_numpy(), r["pred"].to_numpy()),
    "lgbm_nasa": row_nasa(l["rul"].to_numpy(), l["pred"].to_numpy()),
}).reset_index()
out["extra"] = out.lgbm_nasa - out.ridge_nasa

sel = out[out.unit.isin(UNITS)].sort_values(["unit", "seed"])
print(sel.round(2).to_string(index=False))

print("\nPer unit, per seed: total extra NASA")
print(sel.pivot_table(index="unit", columns="seed", values="extra", aggfunc="sum").round(0).to_string())