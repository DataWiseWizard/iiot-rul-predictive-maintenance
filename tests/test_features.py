import pandas as pd

from rul_pm.features.features import add_rolling_means


def _engine(unit=1, values=(1.0, 2.0, 3.0, 4.0, 5.0)):
    return pd.DataFrame({"unit": unit, "cycle": range(1, len(values) + 1), "s": list(values)})


def test_trailing_window_values():
    out = add_rolling_means(_engine(), ["s"], window=3)
    assert out.loc[out.cycle == 1, "s_roll3"].item() == 1.0
    assert out.loc[out.cycle == 2, "s_roll3"].item() == 1.5   # cycles 1,2
    assert out.loc[out.cycle == 3, "s_roll3"].item() == 2.0   # cycles 1,2,3


def test_no_future_leakage():
    base = _engine()
    changed = base.copy()
    changed.loc[changed.cycle == 4, "s"] = 999.0
    a = add_rolling_means(base, ["s"], window=3)
    b = add_rolling_means(changed, ["s"], window=3)
    pd.testing.assert_series_equal(
        a.loc[a.cycle <= 3, "s_roll3"], b.loc[b.cycle <= 3, "s_roll3"]
    )


def test_engines_do_not_mix():
    two = pd.concat(
        [_engine(unit=1), _engine(unit=2, values=(10.0, 10.0, 10.0, 10.0, 10.0))],
        ignore_index=True,
    )
    out = add_rolling_means(two, ["s"], window=3)
    first_of_unit2 = out.loc[(out.unit == 2) & (out.cycle == 1), "s_roll3"].item()
    assert first_of_unit2 == 10.0  # not pulled from unit 1