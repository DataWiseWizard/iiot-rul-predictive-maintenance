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

## D7. Capped RUL label (cap = 125) and scoring convention
Train on rul = min(last_cycle - cycle, 125). Primary scoring: predictions vs
min(true RUL, 125) (option B, matches brief rule c). Secondary, always
reported: predictions vs raw true RUL (option A). 11 of 100 FD001 test
engines have true RUL > 125. Cap fixed before any baseline, never tuned on
test. Rejected: linear label (early-life targets are not learnable from
sensors that still look healthy). Cost: model can never output > 125.

## D8. Feature contract
Built from train only (scripts/build_contract.py -> configs/feature_contract.yaml).
Constant columns (nunique==1) dropped; near-constant kept until ablated.
cycle is optional, decided by ablation in Stage 3. Null: reject. Out of range:
flag. Units unverified. Rejected: imputing nulls in v1 (stuck-sensor risk).
