import numpy as np
import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.evaluation.validation import sample_cuts
from rul_pm.training.tuning import tuned


def _fleet(n_units=20, life=200):
    rows = [(u, c) for u in range(1, n_units + 1) for c in range(1, life + 1)]
    return add_rul(pd.DataFrame(rows, columns=["unit", "cycle"]))


def test_tuned_picks_lower_cost_candidate_and_caches_per_training_set():
    df = _fleet()
    oracle = lambda tr, va: va["rul"].to_numpy()
    late = lambda tr, va: va["rul"].to_numpy() + 20.0
    t = tuned({"late": late, "oracle": oracle})
    cuts = sample_cuts(df, 0)
    preds = t(df, cuts)
    assert np.allclose(preds, cuts["rul"])
    assert list(t.cache.values())[0][0] == "oracle"
    t(df, cuts)
    assert len(t.cache) == 1