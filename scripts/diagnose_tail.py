import numpy as np
import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions
from rul_pm.training.ablation import make_ridge
from rul_pm.training.gbm import make_lgbm

BINS = [0, 20, 40, 60, 80, 100, 126]  # capped truth
BIG = 40                               # "big late" = predicted more than 40 cycles too high
LATE_DIV, EARLY_DIV = 10.0, 13.0       # D4

df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
models = {"ridge_raw": make_ridge(base_cols), "lgbm_raw": make_lgbm(base_cols)}
oof = oof_predictions(df, models)

oof["bias"] = oof["pred"] - oof["rul"]
oof["bin"] = pd.cut(oof["rul"], BINS, right=False)
oof["nasa"] = np.where(oof.bias >= 0,
                       np.exp(oof.bias / LATE_DIV) - 1,
                       np.exp(-oof.bias / EARLY_DIV) - 1)
oof["big_late"] = oof.bias > BIG
oof["big_nasa"] = oof["nasa"].where(oof["big_late"], 0.0)

out = (oof.groupby(["bin", "model"], observed=True)
          .agg(n=("bias", "size"),
               mean_bias=("bias", "mean"),
               p90_bias=("bias", lambda s: s.quantile(0.9)),
               n_big_late=("big_late", "sum"),
               nasa_total=("nasa", "sum"),
               nasa_from_big=("big_nasa", "sum"))
          .round(1))
out["big_share"] = (out["nasa_from_big"] / out["nasa_total"]).round(2)
print(out.to_string())
print("\nNASA totals are summed over 5 seeds; divide by 5 for the per-seed mean.")