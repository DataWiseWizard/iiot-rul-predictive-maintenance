# IIoT Predictive Maintenance: Remaining Useful Life (RUL)

Estimate remaining cycles of turbofan engines (NASA C-MAPSS, simulated data)
so maintenance can be scheduled before failure without replacing healthy parts.

> Status: Stage 0 (scaffold). No results yet.

## Problem
Fleet operators must decide which engine goes to the workshop in the next
slot. Predicting too early wastes usable life; too late risks unplanned
failure. The cost is asymmetric, so RMSE alone is not the success metric.

## Success criteria
See [docs/SUCCESS_CRITERIA.md](docs/SUCCESS_CRITERIA.md), fixed before training.

## Planned scope
Baselines (mean, persistence, Ridge) -> LightGBM with rolling features ->
group-aware validation -> bootstrap CIs, SHAP, uncertainty band ->
MLflow -> FastAPI + Docker -> one Evidently drift report.

## Limitations
Simulated data only. No real-plant validation.
