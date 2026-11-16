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

## D9. Age baseline replaces literal persistence
Persistence (repeat the last RUL) cannot be built at serving time: no earlier
RUL values exist. Replacement: pred = clip(mean_train_lifetime - cycle, 0, 125),
mean lifetime from the training fold only. Changed before any model was run.

## D10. Validation design
I chose random cut points over scoring only the last row because the last row is always RUL 0 and cuts cover the range the test covers, at the cost of a noisier score that depends on a sampling choice, which I reduce with repeated seeds.

## D11. Rolling-mean ablation (Ridge, validation design D10)
Window 10, trailing per-engine means of all contract candidates, computed on full engines before splitting. NASA score (mean over seeds 0–4): raw 1116.27 (std 228.47), rolling 1212.04 (std 342.63). Per-seed differences (rolling − raw): −26.42, +277.92, −39.94, +207.98, +59.33. Late fraction unchanged (0.56); mean late error 19.56 vs 19.41. Decision: Ridge baseline stays on raw features. Rolling features remain a candidate for LightGBM, evaluated under C1 and C2 with the same validation design. Only window 10 was run; any other window must be logged here before it is compared.

## D12. LightGBM comparison (validation design D10, capped truth)
Parameters fixed in training/gbm.py (LGBM_PARAMS): 300 trees, learning rate 0.05, 31 leaves, min child samples 20, subsample 0.8, colsample 0.8, seed 0. Variants: lgbm_raw (contract features) and lgbm_rolling (contract features plus trailing window-10 means, as in D11). Only these two variants are run. Point threshold for C1: 0.9 × Ridge raw mean NASA (1116.27) = 1004.64. C1 also requires a paired-bootstrap 95% CI of the difference, which needs per-engine predictions and will be added before the verdict is recorded. Any tuning later is limited to validation, and every tried setting is logged here.

## D12 result. (validation, capped truth, seeds 0–4)
C1 not met. Point estimates: ridge_raw NASA 1116.27, lgbm_raw 1882.57, lgbm_rolling 2030.63; threshold 1004.64. C2 point estimates met for both LightGBM variants (late fraction 0.51 and 0.47 vs 0.56; mean late error 17.45 and 18.39 vs 19.56); bootstrap CI not yet computed, so C2 is open. Lower MAE and RMSE for lgbm_raw, but higher NASA; primary metric is NASA (D2), so Ridge stays the bar.

## D12 result with paired engine bootstrap (2000 draws, seed 0, 95% CI)
Candidate minus ridge_raw. NASA: lgbm_raw +766.3 [+20.5, +1779.6]; lgbm_rolling +914.4 [+53.2, +2321.3]. C1 not met: point estimate above threshold and CI positive. Late fraction: lgbm_raw −0.052 [−0.086, −0.016]; lgbm_rolling −0.096 [−0.144, −0.048]. Mean late error: lgbm_raw −2.114 [−3.874, −0.419]; lgbm_rolling −1.175 [−3.174, +0.859]. C2 status pending the written rule and the per-variant question. Bootstrap covers engine sampling only; seed std for lgbm_raw NASA is 977.