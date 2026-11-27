"""Asymmetric maintenance cost and tail share (D14).

d = pred - truth. d > 0 is a late prediction.
"""
import numpy as np

TAIL_L = 20.0  # cycles; a late error above this counts as a "big late" cut


def asymmetric_cost(y_true, y_pred, c_late: float = 3.0, c_early: float = 1.0) -> float:
    """Mean cost per cut: c_late * d for late, c_early * |d| for early or exact."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.where(d > 0, c_late * d, c_early * np.abs(d)).mean())


def tail_share(y_true, y_pred, L: float = TAIL_L) -> float:
    """Share of cuts with a late error greater than L cycles."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float((d > L).mean())