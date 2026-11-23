"""Label-budget subsets of the pool (stratified by source, nested: 1% subset is contained in the 10% subset)."""
from __future__ import annotations

import json
import random
from pathlib import Path

SOURCES = ["seadronessee", "visdrone", "floodnet"]


def make_budgets(data_root="data", fracs=(0.01, 0.10), seed=0, out="configs/label_budgets.json"):
    rng = random.Random(seed)
    res = {f"{int(f * 100)}pct": {} for f in fracs}
    for s in SOURCES:
        gt = json.load(open(Path(data_root) / s / "pool" / "annotations.json"))
        ids = [im["id"] for im in gt["images"]]
        rng.shuffle(ids)
        for f in fracs:
            k = max(1, round(f * len(ids)))
            res[f"{int(f * 100)}pct"][s] = sorted(ids[:k])
    Path(out).write_text(json.dumps(res, indent=1))
    return res


def load_budgets(path="configs/label_budgets.json"):
    return json.load(open(path))
