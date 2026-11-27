import pandas as pd
import pytest

from rul_pm.evaluation.gates import (
    evaluate_gates, ratio_sensitivity, select_winner, summarize,
)


def _oof(model, preds, truth=50.0, seeds=range(5)):
    rows = []
    for s in seeds:
        for u, p in enumerate(preds, start=1):
            rows.append({"model": model, "seed": s, "unit": u, "cycle": 1,
                         "rul": truth, "rul_raw": truth, "pred": p})
    return pd.DataFrame(rows)


def _ceilings(cost=1e9, std=1e9, ratio=3.0):
    return {"ratio": ratio, "tail_L": 20.0, "tail_ceiling": 0.05,
            "cost_ceiling": cost, "stability_ceiling": std}


def test_summarize_perfect_model_is_zero():
    s = summarize(_oof("perfect", [50, 50, 50, 50]))
    assert s.loc[0, "mean_cost"] == 0.0
    assert s.loc[0, "tail_share"] == 0.0
    assert s.loc[0, "n_seeds"] == 5


def test_tail_gate_fails_even_with_low_mean_cost():
    # one of 40 engines is a big late error (d = 25): tail share 0.025 is fine,
    # so use 4 engines where one is big late (tail share 0.25 > 0.05)
    df = _oof("tail_bad", [75, 50, 50, 50])
    table = evaluate_gates(df, _ceilings())
    assert bool(table.loc[0, "pass_tail"]) is False
    assert bool(table.loc[0, "passes"]) is False


# def test_derive_ceilings_from_reference():
#     ref = _oof("ridge", [60, 50, 50, 50])  # d = 10 on one engine: cost 30 / 4 = 7.5
#     c = derive_ceilings(ref)
#     assert c["reference_model"] == "ridge"
#     assert c["cost_ceiling"] == pytest.approx(7.5)
#     assert c["stability_ceiling"] == pytest.approx(0.0)


def test_missing_column_raises():
    df = _oof("m", [50, 50]).drop(columns=["pred"])
    with pytest.raises(ValueError):
        summarize(df)


def test_wrong_seeds_raise():
    df = _oof("m", [50, 50], seeds=range(3))
    with pytest.raises(ValueError):
        evaluate_gates(df, _ceilings())


def test_winner_is_lowest_cost_when_clearly_better():
    # 40 engines so the bootstrap difference is not degenerate
    a = _oof("A", [50] * 40)
    b = _oof("B", [60] * 40)  # constant late error of 10 on every engine
    df = pd.concat([a, b], ignore_index=True)
    assert select_winner(df, _ceilings()) == "A"


def test_no_passers_returns_none():
    df = _oof("m", [60, 50, 50, 50])
    assert select_winner(df, _ceilings(cost=0.0)) is None


def test_ratio_sensitivity_uses_default_pass_set():
    df = pd.concat([_oof("A", [50] * 40), _oof("B", [60] * 40)], ignore_index=True)
    out = ratio_sensitivity(df, _ceilings())
    assert set(out["ratio"]) == {2.0, 3.0, 5.0}
    assert (out["best_passer"] == "A").all()