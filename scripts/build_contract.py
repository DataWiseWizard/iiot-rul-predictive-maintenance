from pathlib import Path

import yaml

from rul_pm.data.contract import build_contract
from rul_pm.data.loader import load_raw

contract = build_contract(load_raw("data/raw/train_FD001.txt"))
Path("configs/feature_contract.yaml").write_text(yaml.safe_dump(contract, sort_keys=False))
print("features:", len(contract["features"]), "| dropped:", contract["dropped_constant"])
