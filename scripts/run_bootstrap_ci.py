from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions, paired_bootstrap
from rul_pm.features.features import add_rolling_means
from rul_pm.training.ablation import META, make_ridge
from rul_pm.training.gbm import make_lgbm

WINDOW = 10  # same as D11 and D12

df = add_rul(load_raw("data/raw/train_FD001.txt"))
base_cols = list(df.columns)
sensor_cols = [c for c in base_cols if c not in META]
rolled = add_rolling_means(df, sensor_cols, window=WINDOW)

models = {
    "ridge_raw": make_ridge(base_cols),
    "lgbm_raw": make_lgbm(base_cols),
    "lgbm_rolling": make_lgbm(base_cols, window=WINDOW),
}
oof = oof_predictions(rolled, models)

for cand in ["lgbm_raw", "lgbm_rolling"]:
    for metric in ["nasa", "late_fraction", "mean_late_error"]:
        r = paired_bootstrap(oof, cand, "ridge_raw", metric=metric)
        print(f"{cand} - ridge_raw | {metric:16s} "
              f"diff {r['point']:+10.3f}  95% CI [{r['ci_low']:+10.3f}, {r['ci_high']:+10.3f}]")