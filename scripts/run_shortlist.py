"""D14 shortlist run on FD001 validation only.

    uv run python scripts/run_shortlist.py

Refuses to run unless the gates config and the shortlist are committed and clean,
so the pre-registration is in git history before any result exists.
Use --allow-dirty only for dry runs; its output is not a D14 result.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from rul_pm.data.labels import add_rul
from rul_pm.data.loader import load_raw
from rul_pm.evaluation.bootstrap import oof_predictions
from rul_pm.evaluation.cost import TAIL_L
from rul_pm.evaluation.gates import (
    RATIO_DEFAULT, evaluate_gates, ratio_sensitivity, select_winner,
)
from rul_pm.training.shortlist import SHORTLIST, TUNED

DATA = "data/raw/train_FD001.txt"
CONFIG = Path("configs/d14_gates.json")
GUARDED = [str(CONFIG), "src/rul_pm/training/shortlist.py", "src/rul_pm/training/tuning.py"]
OUT = Path("reports/gates")
KEYS = ("ratio", "tail_L", "tail_ceiling", "cost_ceiling", "stability_ceiling")


def load_config() -> dict:
    cfg = json.loads(CONFIG.read_text())
    bad = [k for k in KEYS if cfg.get(k) is None]
    if bad:
        raise SystemExit(f"{CONFIG} is missing values for {bad}")
    if cfg["ratio"] != RATIO_DEFAULT or cfg["tail_L"] != TAIL_L:
        raise SystemExit("config ratio or tail_L disagrees with the code constants")
    return cfg


def check_committed() -> None:
    out = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED],
                         capture_output=True, text=True, check=True).stdout.strip()
    if out:
        raise SystemExit(f"commit these first (D14 must precede the run):\n{out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    if not args.allow_dirty:
        check_committed()
    cfg = load_config()
    print("config sha256:", hashlib.sha256(CONFIG.read_bytes()).hexdigest())

    df = add_rul(load_raw(DATA))
    oof = oof_predictions(df, SHORTLIST)
    OUT.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OUT / "oof_shortlist.parquet", index=False)

    chosen = {name: [{"best": v[0], "inner_cost": v[1]} for v in fn.cache.values()]
              for name, fn in TUNED.items()}
    (OUT / "chosen_configs.json").write_text(json.dumps(chosen, indent=2))

    table = evaluate_gates(oof, cfg, RATIO_DEFAULT)
    table.to_csv(OUT / "gate_table.csv", index=False)
    sens = ratio_sensitivity(oof, cfg)
    sens.to_csv(OUT / "ratio_sensitivity.csv", index=False)
    winner = select_winner(oof, cfg, RATIO_DEFAULT)

    print(table.to_string(index=False))
    print()
    print(sens.to_string(index=False))
    print(f"\nwinner: {winner}")


if __name__ == "__main__":
    main()