import numpy as np
import pandas as pd

from rul_pm.evaluation.metrics import error_summary
from rul_pm.evaluation.validation import group_folds, sample_cuts

SEEDS = range(5)


def oof_predictions(df, models, n_splits=5, seeds=SEEDS) -> pd.DataFrame:
    """One out-of-fold cut prediction per engine, per seed, per model.

    Same procedure as training.baselines.evaluate, but keeps the rows so
    engine-level resampling is possible.
    """
    folds = group_folds(df, n_splits)
    frames = []
    for seed in seeds:
        parts = {name: [] for name in models}
        for tr_idx, va_idx in folds:
            tr, va = df.iloc[tr_idx], df.iloc[va_idx]
            cuts = sample_cuts(va, seed)
            for name, fn in models.items():
                parts[name].append(cuts.assign(pred=fn(tr, cuts), model=name, seed=seed))
        frames.extend(pd.concat(ps) for ps in parts.values())
    return pd.concat(frames, ignore_index=True)


def paired_bootstrap(oof, candidate, baseline, metric="nasa",
                     n_boot=2000, rng_seed=0, alpha=0.05) -> dict:
    """CI for the seed-averaged difference metric(candidate) - metric(baseline).

    Engines are resampled with replacement, the same engines for both models.
    Negative difference means the candidate is lower (better).
    """
    seeds = sorted(oof["seed"].unique())
    tables = {}
    for s in seeds:
        for m in (candidate, baseline):
            tables[(s, m)] = (
                oof[(oof["seed"] == s) & (oof["model"] == m)]
                .sort_values("unit")
                .reset_index(drop=True)
            )
    units = tables[(seeds[0], candidate)]["unit"].to_numpy()
    for t in tables.values():
        if not np.array_equal(t["unit"].to_numpy(), units):
            raise ValueError("models were not scored on the same engines")

    def mean_metric(idx, m):
        vals = []
        for s in seeds:
            t = tables[(s, m)]
            vals.append(error_summary(t["rul"].to_numpy()[idx],
                                      t["pred"].to_numpy()[idx])[metric])
        return np.mean(vals)

    def diff(idx):
        return mean_metric(idx, candidate) - mean_metric(idx, baseline)

    n = len(units)
    point = diff(np.arange(n))
    rng = np.random.default_rng(rng_seed)
    boots = np.array([diff(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "point": point, "ci_low": lo, "ci_high": hi}