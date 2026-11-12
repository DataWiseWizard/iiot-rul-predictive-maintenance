from pathlib import Path

import pandas as pd

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load a C-MAPSS train/test file and name its 26 columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    return df.astype({"unit": "int32", "cycle": "int32"})


def load_rul(path: str | Path) -> pd.Series:
    """True RUL at each test engine's last cycle; row i belongs to unit i."""
    s = pd.read_csv(path, sep=r"\s+", header=None)[0]
    s.index = pd.RangeIndex(1, len(s) + 1, name="unit")
    s.name = "rul_true"
    return s
