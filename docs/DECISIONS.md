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
