import pandas as pd


def add_rolling_means(df: pd.DataFrame, sensor_cols: list[str], window: int = 3) -> pd.DataFrame:
    """Add trailing per-engine rolling means. Row t uses only rows <= t of the same engine."""
    if window < 1:
        raise ValueError("window must be >= 1")
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    roll = (
        df.groupby("unit")[sensor_cols]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df.join(roll.add_suffix(f"_roll{window}"))