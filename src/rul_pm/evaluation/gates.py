"""D14 gates and ranking on validation out-of-fold predictions (FD001 only).

Input: the frame from rul_pm.evaluation.bootstrap.oof_predictions, with columns
model, seed, unit, cycle, rul (capped truth), rul_raw, pred. One cut per engine per seed.
"""
from __future__ import annotations

import pandas as pd

from rul_pm.evaluation.bootstrap import paired_bootstrap
from rul_pm.evaluation.cost import TAIL_L, asymmetric_cost, tail_share
from rul_pm.evaluation.metrics import nasa_score

REFERENCE_MODEL = "ridge"
RATIO_DEFAULT = 3.0
RATIOS = (2.0, 3.0, 5.0)
TAIL_CEILING = 0.05
REQUIRED = ("model", "seed", "unit", "rul", "pred")
SEEDS = [0, 1, 2, 3, 4]


def check_frame(oof: pd.DataFrame, full: bool = True) -> None:
    missing = set(REQUIRED) - set(oof.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if oof[["rul", "pred"]].isna().any().any():
        raise ValueError("NaN in rul or pred")
    if oof.duplicated(["model", "seed", "unit"]).any():
        raise ValueError("duplicate (model, seed, unit) rows")
    if full and sorted(oof["seed"].unique()) != SEEDS:
        raise ValueError(f"expected seeds {SEEDS}")


def per_seed(oof: pd.DataFrame, ratio: float = RATIO_DEFAULT) -> pd.DataFrame:
    """Mean cost, tail share and NASA per engine for each (model, seed)."""
    check_frame(oof, full=False)
    rows = []
    for (m, s), t in oof.groupby(["model", "seed"]):
        rows.append({
            "model": m,
            "seed": s,
            "mean_cost": asymmetric_cost(t["rul"], t["pred"], c_late=ratio),
            "tail_share": tail_share(t["rul"], t["pred"]),
            "nasa_per_engine": nasa_score(t["rul"], t["pred"]) / len(t),
        })
    return pd.DataFrame(rows)


def summarize(oof: pd.DataFrame, ratio: float = RATIO_DEFAULT) -> pd.DataFrame:
    """Per model: mean cost across seeds, std of that across seeds (ddof=1),
    mean tail share, mean NASA per engine."""
    ps = per_seed(oof, ratio)
    return (
        ps.groupby("model")
        .agg(mean_cost=("mean_cost", "mean"),
             std_cost=("mean_cost", "std"),
             tail_share=("tail_share", "mean"),
             nasa_per_engine=("nasa_per_engine", "mean"),
             n_seeds=("seed", "nunique"))
        .reset_index()
    )


# def derive_ceilings(oof: pd.DataFrame, ratio: float = RATIO_DEFAULT) -> dict:
#     """Cost and stability ceilings from the reference model's OOF predictions.

#     Run once, then frozen to reports/gates/ceilings.json before any shortlist run.
#     See D14 for the caveat that this makes the reference pass by construction.
#     """
#     check_frame(oof)
#     ref = oof[oof["model"] == REFERENCE_MODEL]
#     if ref.empty:
#         raise ValueError(f"no rows for reference model {REFERENCE_MODEL!r}")
#     ps = per_seed(ref, ratio)
#     return {
#         "reference_model": REFERENCE_MODEL,
#         "ratio": ratio,
#         "tail_L": TAIL_L,
#         "tail_ceiling": TAIL_CEILING,
#         "cost_ceiling": float(ps["mean_cost"].mean()),
#         "stability_ceiling": float(ps["mean_cost"].std(ddof=1)),
#         "source": "derived from reference model OOF; see D14",
#     }


def evaluate_gates(oof: pd.DataFrame, ceilings: dict, ratio: float = RATIO_DEFAULT) -> pd.DataFrame:
    """Apply the cost, tail and stability gates. C3 is checked outside this function."""
    check_frame(oof)
    if ceilings["ratio"] != ratio:
        raise ValueError("ceilings were derived at a different cost ratio")
    s = summarize(oof, ratio)
    s["pass_cost"] = s["mean_cost"] <= ceilings["cost_ceiling"]
    s["pass_tail"] = s["tail_share"] <= ceilings["tail_ceiling"]
    s["pass_stability"] = s["std_cost"] <= ceilings["stability_ceiling"]
    s["passes"] = s[["pass_cost", "pass_tail"]].all(axis=1)  # D15: stability is reported, not gated
    return s.sort_values("mean_cost").reset_index(drop=True)


def _cost_fn(ratio: float):
    def fn(y_true, y_pred):
        return asymmetric_cost(y_true, y_pred, c_late=ratio)
    fn.__name__ = f"cost_r{ratio:g}"
    return fn


def select_winner(oof: pd.DataFrame, ceilings: dict, ratio: float = RATIO_DEFAULT):
    """Lowest mean cost among passers. Passers whose paired-bootstrap CI against
    the best passer includes 0 are tied; the tie is broken by tail share, then
    seed std. Returns the winner's name, or None if no model passes."""
    table = evaluate_gates(oof, ceilings, ratio)
    passers = table[table["passes"]]
    if passers.empty:
        return None
    best = passers.iloc[0]["model"]
    tied = [best]
    fn = _cost_fn(ratio)
    for m in passers["model"].iloc[1:]:
        r = paired_bootstrap(oof, m, best, metric=fn)
        if r["ci_low"] <= 0 <= r["ci_high"]:
            tied.append(m)
    pool = passers[passers["model"].isin(tied)].sort_values(["tail_share", "std_cost"])
    return pool.iloc[0]["model"]


def ratio_sensitivity(oof: pd.DataFrame, ceilings: dict,
                      ratio_default: float = RATIO_DEFAULT,
                      ratios=RATIOS) -> pd.DataFrame:
    """Best passer at each cost ratio. The pass set is fixed at the default ratio;
    ratios only change the ranking, not the gates."""
    table = evaluate_gates(oof, ceilings, ratio_default)
    pass_set = table.loc[table["passes"], "model"].tolist()
    rows = []
    for r in ratios:
        s = summarize(oof, r)
        s = s[s["model"].isin(pass_set)].sort_values("mean_cost")
        rows.append({"ratio": r, "best_passer": s.iloc[0]["model"] if len(s) else None})
    return pd.DataFrame(rows)