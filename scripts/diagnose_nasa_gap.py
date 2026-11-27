import numpy as np
import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions
from rul_pm.evaluation.metrics import error_summary
from rul_pm.training.ablation import make_ridge
from rul_pm.training.gbm import make_lgbm

LATE_DIV, EARLY_DIV = 10.0, 13.0  # D4
TOP_K = 10


def row_nasa(truth, pred):
    d = pred - truth
    return np.where(d >= 0, np.exp(d / LATE_DIV) - 1, np.exp(-d / EARLY_DIV) - 1)


df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
models = {"ridge_raw": make_ridge(base_cols), "lgbm_raw": make_lgbm(base_cols)}
oof = oof_predictions(df, models)

# Check: per-row contributions must sum to the official NASA score
for (model, seed), t in oof.groupby(["model", "seed"]):
    c = row_nasa(t["rul"].to_numpy(), t["pred"].to_numpy())
    total = error_summary(t["rul"], t["pred"])["nasa"]
    assert np.isclose(c.sum(), total), f"{model} seed {seed} mismatch"

# Pair the two models on identical cuts, per seed
frames = []
for seed in sorted(oof["seed"].unique()):
    r = oof[(oof.seed == seed) & (oof.model == "ridge_raw")].set_index(["unit", "cycle"])
    l = oof[(oof.seed == seed) & (oof.model == "lgbm_raw")].set_index(["unit", "cycle"]).loc[r.index]
    c_r = row_nasa(r["rul"].to_numpy(), r["pred"].to_numpy())
    c_l = row_nasa(l["rul"].to_numpy(), l["pred"].to_numpy())
    frames.append(pd.DataFrame({
        "seed": seed,
        "unit": r.index.get_level_values("unit"),
        "rul": r["rul"].to_numpy(),
        "pred_ridge": r["pred"].to_numpy(),
        "pred_lgbm": l["pred"].to_numpy(),
        "extra": c_l - c_r,
    }))
cuts = pd.concat(frames, ignore_index=True)
cuts["lgbm_late"] = cuts.pred_lgbm >= cuts.rul
cuts["ridge_bias"] = cuts.pred_ridge - cuts.rul
cuts["lgbm_bias"] = cuts.pred_lgbm - cuts.rul

# 1. Top-K LightGBM-loss cuts per seed
top = pd.concat([g.nlargest(TOP_K, "extra") for _, g in cuts.groupby("seed")])
print("Top 10 LightGBM-loss cuts per seed (unit:extra NASA):")
for seed, g in top.groupby("seed"):
    print(f"  seed {seed}: " + ", ".join(f"{u}:{e:.0f}" for u, e in zip(g.unit, g.extra)))

# 2. Units that appear in the top K across several seeds
count = top.groupby("unit")["seed"].nunique().sort_values(ascending=False)
print("\nUnits in the top 10 for at least 2 seeds (count of seeds):")
print(count[count >= 2].to_string())

# 3. Per-unit summary across all seeds
by_unit = cuts.groupby("unit").agg(
    extra_sum=("extra", "sum"),
    lgbm_late_frac=("lgbm_late", "mean"),
    ridge_bias=("ridge_bias", "mean"),
    lgbm_bias=("lgbm_bias", "mean"),
    rul_mean=("rul", "mean"),
).sort_values("extra_sum", ascending=False)
print(f"\nTotal extra NASA over all seeds: {by_unit.extra_sum.sum():.1f}")
pos = by_unit.extra_sum.clip(lower=0)
print(f"Share of positive extra from top {TOP_K} units: {pos.nlargest(TOP_K).sum() / pos.sum():.2f}")
print("\nPer-unit summary, top 10 by extra NASA (bias = predicted minus true RUL):")
print(by_unit.head(TOP_K).round(2).to_string())