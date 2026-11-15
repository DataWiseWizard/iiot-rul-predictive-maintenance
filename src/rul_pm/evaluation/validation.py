import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

CUT_LO, CUT_HI = 10, 150  # raw-RUL range of simulated cuts, from the PHM08 paper


def group_folds(df: pd.DataFrame, n_splits: int = 5) -> list:
    """(train_idx, val_idx) pairs; all rows of an engine stay on one side."""
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df["unit"]))


def sample_cuts(df: pd.DataFrame, seed: int, lo: int = CUT_LO, hi: int = CUT_HI) -> pd.DataFrame:
    """One row per engine at a random cut point (simulated censoring).

    Needs rul_raw/rul columns from add_rul. Raw RUL at the cut is uniform in
    [lo, min(hi, lifetime - 1)].
    """
    rng = np.random.default_rng(seed)
    life = df.groupby("unit")["cycle"].max()
    max_rul = np.minimum(hi, life.to_numpy() - 1)
    if (max_rul < lo).any():
        raise ValueError("an engine is too short for the requested cut range")
    cut_rul = rng.integers(lo, max_rul + 1)
    cuts = pd.DataFrame({"unit": life.index, "cycle": life.to_numpy() - cut_rul})
    return df.merge(cuts, on=["unit", "cycle"], how="inner")
