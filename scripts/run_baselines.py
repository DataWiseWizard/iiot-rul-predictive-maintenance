from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.training.baselines import evaluate

res = evaluate(add_rul(load_raw("data/raw/train_FD001.txt")))
cols = ["nasa", "mae", "rmse", "late_fraction", "mean_late_error"]
g = res.groupby(["truth", "model"])[cols]
print(g.mean().round(2).to_string())
print("\nstd over seeds:")
print(g.std().round(2).to_string())
