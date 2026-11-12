from pathlib import Path

import pytest

from rul_pm.data.loader import COLUMNS, load_raw, load_rul

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
needs_data = pytest.mark.skipif(
    not (RAW / "train_FD001.txt").exists(), reason="raw data not downloaded"
)


def _line(unit, cycle):
    return " ".join([str(unit), str(cycle)] + ["0.5"] * 24) + " \n"


def test_loader_names_columns(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text(_line(1, 1) + _line(1, 2))
    df = load_raw(f)
    assert list(df.columns) == COLUMNS
    assert len(df) == 2


def test_loader_rejects_wrong_column_count(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("1 1 0.5\n")
    with pytest.raises(ValueError):
        load_raw(f)


@needs_data
def test_fd001_matches_readme_counts():
    train = load_raw(RAW / "train_FD001.txt")
    test = load_raw(RAW / "test_FD001.txt")
    assert train["unit"].nunique() == 100
    assert test["unit"].nunique() == 100
    assert len(load_rul(RAW / "RUL_FD001.txt")) == 100
    assert (len(train), len(test)) == (20631, 13096)


@needs_data
def test_cycles_contiguous_from_one_and_no_nulls():
    for name in ("train", "test"):
        df = load_raw(RAW / f"{name}_FD001.txt")
        g = df.groupby("unit")["cycle"]
        assert (g.min() == 1).all()
        assert (g.max() == g.count()).all()
        assert not df.isna().any().any()
