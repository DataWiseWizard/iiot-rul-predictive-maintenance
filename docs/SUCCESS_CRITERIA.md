# Success criteria (fixed before any model training)

Data: NASA C-MAPSS FD001 (simulated). Evaluation: one prediction per test
engine at its last observed cycle (convention to be verified in Stage 1).
Error sign: d = predicted RUL - true RUL. d > 0 is late, d < 0 is early.

Primary metric: NASA score. Secondary: MAE, RMSE, late fraction, mean late error.
All comparisons use a paired bootstrap over engines, 95% CI.

C1. LightGBM beats constant-mean, persistence and Ridge on the primary
    metric. The CI of the difference excludes 0 in LightGBM's favour, and the
    improvement over the best baseline is >= __% (set after Stage 2 baselines,
    before any LightGBM run; the commit history is the evidence).
C2. Late-prediction safety: the model's late fraction is <= the best
    baseline's late fraction AND its mean late error is <= the best
    baseline's, each reported with a bootstrap 95% CI.
C3. `make check` (ruff + pytest, including leakage and group-split guards)
    passes locally and in CI on a fresh clone.

Anything not met is reported as not met. No claim of real plant validation
or real downtime savings: the data is simulated.
