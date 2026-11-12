import pandas as pd

from rul_pm.data.contract import NEVER_FEATURES, build_contract
from rul_pm.data.loader import COLUMNS


def _df():
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in COLUMNS})
    df["s1"] = 5.0
    df["setting3"] = 100.0
    return df


def test_constant_columns_dropped():
    c = build_contract(_df())
    assert {"s1", "setting3"} <= set(c["dropped_constant"])
    assert "s1" not in c["features"]


def test_ids_labels_and_cycle_never_in_features():
    c = build_contract(_df())
    for banned in NEVER_FEATURES + ["cycle"]:
        assert banned not in c["features"]
