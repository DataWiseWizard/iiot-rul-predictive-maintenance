import numpy as np

A_EARLY = 13.0  # divisor for early predictions (d < 0), see D4
A_LATE = 10.0   # divisor for late predictions (d >= 0)


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.where(d < 0, np.exp(-d / A_EARLY) - 1, np.exp(d / A_LATE) - 1).sum())


def error_summary(y_true, y_pred) -> dict:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    late = d > 0
    return {
        "nasa": nasa_score(y_true, y_pred),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "late_fraction": float(late.mean()),
        "mean_late_error": float(d[late].mean()) if late.any() else 0.0,
    }
