import pandas as pd

from rul_pm.data.loader import SENSOR_COLS, SETTING_COLS

CANDIDATES = SETTING_COLS + SENSOR_COLS
NEVER_FEATURES = ["unit", "rul", "rul_raw"]  # id and labels, rule (f)


def build_contract(train: pd.DataFrame) -> dict:
    """Derive the feature contract from TRAIN data only."""
    constant = [c for c in CANDIDATES if train[c].nunique() == 1]
    features = [c for c in CANDIDATES if c not in constant]
    return {
        "features": features,
        "optional_features": ["cycle"],  # decided by ablation in Stage 3
        "dropped_constant": constant,
        "excluded_always": NEVER_FEATURES,
        "ranges_train": {c: [float(train[c].min()), float(train[c].max())] for c in features},
        "dtype": "float64",
        "null_policy": "reject",
        "out_of_range_policy": "flag",
        "units": "unverified: paper Table 2 gives units but not column order",
    }
