from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.features.features import add_rolling_means
from rul_pm.training.ablation import META, make_ridge
from rul_pm.training.baselines import evaluate

# Fixed before running. Do not change after seeing results.
WINDOW = 10

df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
sensor_cols = [c for c in base_cols if c not in META]
rolled = add_rolling_means(df, sensor_cols, window=WINDOW)

models = {
    "ridge_raw": make_ridge(base_cols),
    "ridge_rolling": make_ridge(base_cols, window=WINDOW),
}
res = evaluate(rolled, models=models)

cols = ["nasa", "mae", "rmse", "late_fraction", "mean_late_error"]
g = res.groupby(["truth", "model"])[cols]
print(g.mean().round(2).to_string())
print("\nstd over seeds:")
print(g.std().round(2).to_string())

nasa = res[res.truth == "rul"].pivot(index="seed", columns="model", values="nasa")
diff = nasa["ridge_rolling"] - nasa["ridge_raw"]
print("\nper-seed NASA difference (rolling - raw, negative = rolling better):")
print(diff.round(2).to_string())