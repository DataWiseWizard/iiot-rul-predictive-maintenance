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

## C2 verdict (validation, capped truth, paired engine bootstrap, 95% CI) 
The C2 rule was written after the D12 LightGBM run and is not a pre-registration. Reference values from docs/BASELINES.md (recorded before LightGBM): Ridge late fraction 0.56, mean late error 19.6. Verdict per variant: lgbm_raw met (late fraction 0.51, CI of difference [−0.086, −0.016]; mean late error 17.45, CI [−3.874, −0.419]). lgbm_rolling met on point estimates (late fraction 0.47, CI [−0.144, −0.048]; mean late error 18.39, CI [−3.174, +0.859], not distinguishable from no change). C1 remains not met, so the overall LightGBM result is a fail on the primary criterion.

## Diagnostic (seed 0, not pre-registered) 
LightGBM’s NASA total is more concentrated than Ridge’s: the top 10 cuts give 83% vs 56%. The gap is late-biased (late share 0.85 vs 0.77). Units 28 and 84 make up 124 of LightGBM’s extra NASA on seed 0, but seed 0 alone is better for LightGBM (−95.7), so the mean gap comes from other seeds. Cross-seed check pending.

## D14: Success criteria revision, shortlist and nested config selection (FD001)

**Date:** 2026-11-27
**Status:** Pre-registered and run. Outcome: no model passed all gates.
**Replaces:** C1 (beat Ridge by 10% on NASA) and C2 (late fraction and mean late error no worse than Ridge). Both are withdrawn. C2 was originally written after the D12 run and was recorded as post-hoc.
**Supersedes:** D13 (LightGBM regularization grid), which was never run. Its idea is folded into the LightGBM grid below.

### Why
Pass/fail defined relative to Ridge made the verdict depend on one baseline, and it would be meaningless once Ridge became the candidate. Criteria are now absolute thresholds, and passing models are then ranked.

### Primary metric
Mean asymmetric linear cost per cut, with error d = pred - true and d > 0 late:
cost = c_late * d if d > 0, else c_early * |d|, with c_late / c_early = 3 and c_early = 1.
Sensitivity is reported at ratios 2 and 5. NASA per engine (D4 definition, unchanged) is reported alongside in every table.

**Post-hoc declaration:** the switch from NASA to linear cost was made after the D12 results, in which LightGBM's NASA was significantly worse than Ridge's. Under linear cost, rough estimates backed out of the reported D12 summary statistics were about 40 per cut for Ridge and about 33 for LightGBM raw (estimates, not runs). The metric change is therefore recorded as post-hoc.

### Gates (validation, FD001, capped truth, seeds 0-4)
- Cost gate: mean cost per cut <= 30. Meaning: no worse than being 10 cycles late on every cut.
- Tail gate: share of cuts with late error > 20 cycles <= 0.05.
- Stability gate: seed std of mean cost <= 3.0. (Made reported-only by D15.)
- Reproducibility gate (C3): `make check` passes on a fresh clone, locally and in CI. Not yet verified.

Ceilings are hand-written policy values in `configs/d14_gates.json` (sha256 ea86dd2bde66415093dcce9e0ed7ec97c070de41f0f468e3f7b96e36a3b8ba7d). They are not derived from any model run. They were written with rough knowledge of the cost estimates above.

### Shortlist and grids (fixed before the run)
Config selection is nested: each tuned model chooses its config using only the training engines of each outer fold (inner 3-fold GroupKFold, inner cut seeds 100-102, selection by mean cost at ratio 3), then refits on the full training fold. Selection is cached per outer fold.
- ridge: alpha 1.0, untuned, continuity row for D12
- ridge_tuned: alpha in {1, 10, 100}
- elastic_net_tuned: (alpha, l1_ratio) in {(0.01, 0.5), (0.1, 0.5), (0.1, 0.9)}
- extra_trees_tuned: 300 trees, (min_samples_leaf, max_features) in {(5, 1.0), (20, 1.0), (20, 0.5)}
- lgbm_quantile_tuned: quantile alpha 0.35, (num_leaves, min_child_samples) in {(31, 20), (15, 100), (7, 400)}, other parameters as D12
- blend_tuned: 0.5 * ridge_tuned + 0.5 * extra_trees_tuned
Windowed models (MLP, 1D CNN, GRU) were deferred: they need a data-contract note.

### Ranking and rules
Among models passing all gates: lowest mean cost at ratio 3. Models whose paired-bootstrap 95% CI against the best passer includes 0 are tied; ties are broken by tail share, then seed std. If no model passes, the result is reported as is and the gates are not relaxed. The test set is used once.

### Result (validation, mean over seeds 0-4)
| model | mean cost | seed std | tail share | NASA / engine | passes |
|---|---|---|---|---|---|
| lgbm_quantile_tuned | 29.71 | 4.54 | 0.136 | 15.29 | no (tail, stability) |
| extra_trees_tuned | 33.37 | 5.13 | 0.188 | 15.83 | no |
| blend_tuned | 35.95 | 4.63 | 0.202 | 11.11 | no |
| ridge_tuned | 40.21 | 4.87 | 0.258 | 11.16 | no |
| ridge | 40.21 | 4.87 | 0.258 | 11.16 | no |
| elastic_net_tuned | 40.22 | 4.88 | 0.258 | 11.16 | no |

**Outcome: no model passed all gates. Winner: none.**
- Only lgbm_quantile_tuned passed the cost gate, by 0.29, which is small against a seed std of 4.54. No paired test was run on this margin.
- Every model failed the tail gate and the stability gate.
- The ridge row reproduces the D12 Ridge NASA (11.1627 per engine, 1116.27 over 100 engines), which checks the new pipeline against the old.
- ridge, ridge_tuned and elastic_net_tuned are practically the same model (cost 40.21 to 40.22): tuning did not move the linear family.
- Cost and NASA rank the models differently. Tree models rank best on cost and tail share; the linear models and the blend rank best on NASA.

### Diagnostic (not pre-registered)
NASA split into early and late parts, from saved out-of-fold predictions:
| model | early share of NASA | mean early err | p95 early err | mean late err | p95 late err |
|---|---|---|---|---|---|
| blend_tuned | 0.176 | 14.25 | 37.54 | 17.86 | 44.30 |
| elastic_net_tuned | 0.222 | 17.82 | 41.27 | 18.39 | 42.68 |
| extra_trees_tuned | 0.114 | 12.77 | 35.20 | 17.31 | 49.15 |
| lgbm_quantile_tuned | 0.249 | 14.36 | 43.99 | 16.91 | 48.76 |
| ridge | 0.222 | 17.82 | 41.28 | 18.38 | 42.68 |
| ridge_tuned | 0.222 | 17.83 | 41.28 | 18.38 | 42.68 |
- Withdrawn: the hypothesis that tree models lose on NASA because of larger early errors. Early errors are 11% to 25% of NASA for every model.
- Unverified hypothesis: tree models reduce moderate late errors but produce more extreme ones (p95 late error about 49 vs 43 for Ridge), which the exponential NASA term punishes. Only one quantile was examined.
- The tail gate (L = 20) counts moderate and extreme late errors equally and cannot see this. It is not changed here.

### Caveats
- The 5 seeds reuse the same 100 engines, so seeds are not independent samples.
- The stability measure is shared-cut noise (see D15).
- Tuned configs per fold are saved in `reports/gates/chosen_configs.json` and have not been reviewed.

### Reproduce
`uv run python scripts/run_shortlist.py` (refuses to run unless the config, `shortlist.py` and `tuning.py` are committed and clean). Outputs in `reports/gates/`.

---

## D15: Stability gate made reported-only; windowed models next; fallback rule

**Date:** 2026-11-27
**Status:** Approved. Post-hoc revision of D14, declared as such.

### Changes
1. The stability gate is removed as a pass/fail condition. The seed std of mean cost is still computed and reported.
2. The cost gate (<= 30) and the tail gate (<= 0.05 share of cuts late by more than 20 cycles) are unchanged.
3. The deferred windowed models (MLP, 1D CNN, GRU) will be run under the unchanged cost and tail gates. Their window data contract and grids will be pre-registered in D16 and committed before the run.
4. Fallback: if no model passes after the windowed run, the model with the lowest mean cost at ratio 3 (ties as in D14) is carried forward as "best available, gates not met". The README states this plainly. The test set is used once, on that model only.

### Reason for 1
Within a seed, every model is scored on the same random cuts, so the across-seed std mostly measures how hard that seed's cuts are. The D14 results show it: all models had a seed std between 4.5 and 5.1. The gate could not discriminate between models, and the 3.0 ceiling was below the protocol's noise floor. This was a design error in D14.

### Why this is not outcome-driven
No model passed D14, and none passes after this change, because the tail gate still fails every model. The revision does not change the pass set.

### Not pursued
Lowering the LightGBM quantile alpha to reduce tail share would amount to tuning toward the gate on the same validation data. If wanted, it needs its own pre-registered decision.

### Post-hoc items now on record
C2 (written after D12), the switch of primary metric to linear cost (D14), and this revision (D15).