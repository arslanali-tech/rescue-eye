"""Zero-shot open-vocabulary benchmark: detectors x inference passes (full image vs SAHI slices) x prompt sets,
evaluated on the held-out test split of every source."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache as C
from .budget import SOURCES
from .evaluate import coco_eval
from .pipeline import PostCfg, apply_all

DETECTORS = {
    "yoloworld": {"label": "YOLO-World-v2-S", "full": "full640"},
    "owlv2": {"label": "OWLv2-B/16", "full": "full"},
    "gdino": {"label": "Grounding-DINO-T", "full": "full"},
}


def available_passes(det, split="test"):
    out = {}
    for s in SOURCES:
        for p in sorted((C.CACHE / det).glob(f"{s}_{split}_*.pkl")):
            out.setdefault(p.stem.split("_", 2)[2], set()).add(s)
    return sorted(k for k, v in out.items() if v == set(SOURCES))


def load_gt(source, split="test", data_root="data"):
    return json.load(open(Path(data_root) / source / split / "annotations.json"))


def evaluate_cfg(det, cfg: PostCfg, split="test", caches=None, gts=None, image_ids=None):
    rows = []
    for s in SOURCES:
        cb = {p: (caches or {}).get((det, s, p)) or C.load(det, s, split, p) for p in cfg.passes}
        raws = {p: v["raw"] for p, v in cb.items()}
        ids = image_ids.get(s) if image_ids else None
        gt = (gts or {}).get(s) or load_gt(s, split)
        if ids is not None:
            keep = set(ids)
            gt = {**gt, "images": [im for im in gt["images"] if im["id"] in keep],
                  "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
        dets = apply_all(raws, cfg, [im["id"] for im in gt["images"]])
        m = coco_eval(gt, dets, s)
        sec = sum(float(np.mean([cb[p]["sec"][im["id"]] for im in gt["images"]])) for p in cfg.passes)
        rows.append({"source": s, **m, "sec_per_image": sec})
    return rows


def run(out_dir="outputs/results"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caches, gts = {}, {s: load_gt(s) for s in SOURCES}
    records = []
    for det, meta in DETECTORS.items():
        passes = available_passes(det)
        if not passes:
            continue
        for p in passes:
            for s in SOURCES:
                caches[(det, s, p)] = C.load(det, s, "test", p)
        full = meta["full"] if meta["full"] in passes else passes[0]
        sliced = [p for p in passes if p.startswith("s")]
        combos = [(p,) for p in passes] + [(full, p) for p in sliced]
        for combo in combos:
            for merge in (["nms", "nmm"] if len(combo) > 1 or combo[0].startswith("s") else ["nms"]):
                cfg = PostCfg(prompt_set="specific", passes=combo, merge=merge, merge_thr=0.5)
                for r in evaluate_cfg(det, cfg, caches=caches, gts=gts):
                    records.append({"detector": meta["label"], "passes": "+".join(combo), "merge": merge,
                                    "prompt_set": "specific", **r})
        # prompt ablation on the full-image pass and on every sliced pass (+full)
        for combo in [(full,)] + [(full, p) for p in sliced]:
            for ps in ["generic", "context", "ensemble"]:
                cfg = PostCfg(prompt_set=ps, passes=combo, merge="nms", merge_thr=0.5)
                for r in evaluate_cfg(det, cfg, caches=caches, gts=gts):
                    records.append({"detector": meta["label"], "passes": "+".join(combo), "merge": "nms",
                                    "prompt_set": ps, **r})
        print(det, "done", len(records))
    df = pd.DataFrame(records)
    df.to_csv(out / "zeroshot_all.csv", index=False)
    return df


def summarise(df: pd.DataFrame):
    """Macro-average over the three sources (each source weighted equally)."""
    keys = ["detector", "passes", "merge", "prompt_set"]
    agg = df.groupby(keys).agg(AP50=("AP50", "mean"), AP50_95=("AP50_95", "mean"),
                               AP50_small=("AP50_small", "mean"), R50_small=("R50_small", "mean"),
                               sec_per_image=("sec_per_image", "mean")).reset_index()
    return agg.sort_values("AP50", ascending=False)


def cli_zeroshot(cfg, argv):
    df = run()
    s = summarise(df)
    s.to_csv("outputs/results/zeroshot_summary.csv", index=False)
    print(s.head(30).to_string(index=False))
