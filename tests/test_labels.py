import pandas as pd

from rul_pm.data.labels import RUL_CAP, add_rul, cap_truth


def _df():
    return pd.DataFrame(
        {"unit": [1] * 200 + [2] * 3, "cycle": list(range(1, 201)) + [1, 2, 3]}
    )


def test_last_row_has_zero_rul():
    out = add_rul(_df())
    assert (out.groupby("unit")["rul_raw"].min() == 0).all()


def test_labels_for_200_cycle_engine():
    e = add_rul(_df()).query("unit == 1").set_index("cycle")
    assert (e.loc[1, "rul_raw"], e.loc[1, "rul"]) == (199, 125)
    assert e.loc[75, "rul"] == 125
    assert e.loc[76, "rul"] == 124
    assert e.loc[150, "rul"] == 50


def test_cap_never_exceeded_and_short_engine_untouched():
    out = add_rul(_df())
    assert out["rul"].max() <= RUL_CAP
    assert out.query("unit == 2")["rul"].tolist() == [2, 1, 0]


def test_cap_truth_uses_same_cap():
    assert cap_truth(pd.Series([7, 125, 145])).tolist() == [7, 125, 125]
