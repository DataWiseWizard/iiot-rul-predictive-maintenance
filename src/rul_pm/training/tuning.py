"""Nested config selection (D14): choose a config using only the training engines."""
import numpy as np
import pandas as pd

from rul_pm.evaluation.cost import asymmetric_cost
from rul_pm.evaluation.validation import group_folds, sample_cuts

INNER_SPLITS = 3
INNER_SEEDS = (100, 101, 102)  # disjoint from the outer seeds 0-4
SELECT_RATIO = 3.0


def tuned(candidates: dict, inner_splits=INNER_SPLITS, inner_seeds=INNER_SEEDS,
          ratio=SELECT_RATIO):
    """Wrap {name: predict(tr, va)} into one predict(tr, va).

    The best candidate (lowest mean cost on inner-validation cuts) is chosen
    from `tr` alone, cached per training set, then refit on all of `tr`.
    Ties go to the first candidate in dict order.
    """
    cache = {}

    def select(tr):
        tr = tr.reset_index(drop=True)
        folds = group_folds(tr, inner_splits)
        scores = {}
        for name, fn in candidates.items():
            ys, ps = [], []
            for a, b in folds:
                va = tr.iloc[b]
                cuts = pd.concat([sample_cuts(va, s) for s in inner_seeds],
                                 ignore_index=True)
                ps.append(np.asarray(fn(tr.iloc[a], cuts)))
                ys.append(cuts["rul"].to_numpy())
            scores[name] = asymmetric_cost(np.concatenate(ys), np.concatenate(ps),
                                           c_late=ratio)
        return min(scores, key=scores.get), scores

    def predict(tr, va):
        key = tuple(sorted(tr["unit"].unique()))
        if key not in cache:
            cache[key] = select(tr)
        return candidates[cache[key][0]](tr, va)

    predict.cache = cache
    return predict 