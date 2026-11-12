# Decisions

## D1. uv over venv+pip
uv writes a lockfile and pins the Python version, so a clean machine
reproduces the exact environment. Cost: contributors need uv installed.

## D2. NASA score as the primary metric
It encodes the early/late cost asymmetry. Cost: it is exponential, so a few
badly late engines dominate it, which is noisy with ~100 test engines. We
mitigate with bootstrap CIs and report secondary metrics.

## D3. Python 3.12 pinned
LightGBM, SHAP, MLflow and Evidently wheel support on 3.14 is unverified.

## D4. NASA score constants: open discrepancy
Paper text says late predictions are penalised more. Printed eq. 11 pairs
a1=10 with d<0 (early) and a2=13 with d>=0 (late), which says the opposite.
Working assumption: late divisor 10, early divisor 13, matching the text.
To be settled by Figure 9 and a unit test before Stage 4.

### D4 resolution
Figure 9 of the PHM08 paper shows late errors (+50) scoring higher than
early (-50). Convention used: d = pred - true; d<0: exp(-d/13)-1;
d>=0: exp(d/10)-1. Printed eq. 11 appears to have the constants swapped.
To be locked by a unit test in Stage 4.

## D5. Pandas over Polars
(sentence from Rudraksha goes here)

## D6. Evaluation convention (FD001)
One prediction per test engine, at its last observed cycle, scored against
RUL_FD001.txt. Evidence: last cycle + RUL has the same distribution as
train lifetimes (mean 206.5 vs 206.3). Paper implies it; not stated outright.
