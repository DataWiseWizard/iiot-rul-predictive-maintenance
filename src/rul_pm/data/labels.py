import pandas as pd

RUL_CAP = 125


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add rul_raw (last_cycle - cycle) and rul (clipped at cap).

    TRAIN DATA ONLY: assumes each engine runs to failure. On the truncated
    test files the last row is NOT the failure point, so this would be wrong.
    """
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = last - out["cycle"]
    out["rul"] = out["rul_raw"].clip(upper=cap)
    return out


def cap_truth(rul_true: pd.Series, cap: int = RUL_CAP) -> pd.Series:
    """Apply the same cap to the test answer key (scoring option B)."""
    return rul_true.clip(upper=cap)
