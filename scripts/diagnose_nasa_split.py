"""Split each model's NASA into early and late parts, from saved OOF predictions (no new fits)."""
import numpy as np
import pandas as pd

from rul_pm.evaluation.metrics import A_EARLY, A_LATE

oof = pd.read_parquet("reports/gates/oof_shortlist.parquet")
d = oof["pred"] - oof["rul"]
oof = oof.assign(
    d=d,
    nasa_part=np.where(d < 0, np.exp(-d / A_EARLY) - 1, np.exp(d / A_LATE) - 1),
)
rows = []
for m, t in oof.groupby("model"):
    early, late = t[t["d"] < 0], t[t["d"] >= 0]
    rows.append({
        "model": m,
        "nasa_early_share": early["nasa_part"].sum() / t["nasa_part"].sum(),
        "mean_early_err": -early["d"].mean(),
        "p95_early_err": -early["d"].quantile(0.05),
        "mean_late_err": late["d"].mean(),
        "p95_late_err": late["d"].quantile(0.95),
        "share_early_gt30": (t["d"] < -30).mean(),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))