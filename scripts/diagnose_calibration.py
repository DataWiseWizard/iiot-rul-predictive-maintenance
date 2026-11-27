import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions
from rul_pm.training.ablation import make_ridge
from rul_pm.training.gbm import make_lgbm

BINS = [0, 20, 40, 60, 80, 100, 126]  # capped truth; last bin includes 125

df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
models = {"ridge_raw": make_ridge(base_cols), "lgbm_raw": make_lgbm(base_cols)}
oof = oof_predictions(df, models)

oof["bin"] = pd.cut(oof["rul"], BINS, right=False)
oof["bias"] = oof["pred"] - oof["rul"]
out = (oof.groupby(["bin", "model"], observed=True)
          .agg(n=("pred", "size"), mean_true=("rul", "mean"),
               mean_pred=("pred", "mean"), mean_bias=("bias", "mean"))
          .round(1))
print(out.to_string())