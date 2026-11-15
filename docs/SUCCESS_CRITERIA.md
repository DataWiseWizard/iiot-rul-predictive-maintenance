# Success criteria (fixed before any model training)

Data: NASA C-MAPSS FD001 (simulated). Evaluation: one prediction per test
engine at its last observed cycle (convention to be verified in Stage 1).
Error sign: d = predicted RUL - true RUL. d > 0 is late, d < 0 is early.

Primary metric: NASA score. Secondary: MAE, RMSE, late fraction, mean late error.
All comparisons use a paired bootstrap over engines, 95% CI.

C1. LightGBM must beat the best baseline on the NASA score by at least 10%. “Best baseline” is Ridge, scored on the
    validation design in D10 with capped truth. The comparison uses the mean NASA score over seeds 0–4. The threshold is 0.9 × the Ridge mean, which is about 1,004 at the current baseline run. The paired-bootstrap 95% CI of the difference must also exclude 0.
C2. Best baseline for C2 is the same Ridge model as C1, using the same validation design, seeds, and capped truth.
C3. `make check` (ruff + pytest, including leakage and group-split guards)
    passes locally and in CI on a fresh clone.

Anything not met is reported as not met. No claim of real plant validation
or real downtime savings: the data is simulated.
