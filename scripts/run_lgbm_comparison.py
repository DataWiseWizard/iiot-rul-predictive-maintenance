from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.features.features import add_rolling_means
from rul_pm.training.ablation import META, make_ridge
from rul_pm.training.baselines import evaluate
from rul_pm.training.gbm import make_lgbm

WINDOW = 10  # same as D11

df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
sensor_cols = [c for c in base_cols if c not in META]
rolled = add_rolling_means(df, sensor_cols, window=WINDOW)

models = {
    "ridge_raw": make_ridge(base_cols),
    "lgbm_raw": make_lgbm(base_cols),
    "lgbm_rolling": make_lgbm(base_cols, window=WINDOW),
}
res = evaluate(rolled, models=models)

cols = ["nasa", "mae", "rmse", "late_fraction", "mean_late_error"]
g = res.groupby(["truth", "model"])[cols]
print(g.mean().round(2).to_string())
print("\nstd over seeds:")
print(g.std().round(2).to_string())

nasa = res[res.truth == "rul"]
ridge_mean = nasa[nasa.model == "ridge_raw"].nasa.mean()
threshold = 0.9 * ridge_mean
print(f"\nC1 point threshold (0.9 x ridge_raw mean): {threshold:.2f}")
for m in ["lgbm_raw", "lgbm_rolling"]:
    mean = nasa[nasa.model == m].nasa.mean()
    verdict = "under threshold" if mean <= threshold else "over threshold"
    print(f"{m}: mean NASA {mean:.2f} ({verdict})")