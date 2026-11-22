"""Run a detector once per (image, pass) and cache its raw phrase-level outputs.

A *pass* is either ``full<imgsz>`` (whole image) or ``s<size>`` (SAHI slices of ``size`` px, 20% overlap).
Raw outputs are unmerged and keep one score per text phrase, so prompt sets, score thresholds, slice fusion
and NMS/NMM can all be changed later without touching the network again. This is what makes the prompt
ablation and the surrogate-assisted search cheap.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .detectors import PHRASES, RawDet, build
from .slicing import slice_windows

CACHE = Path("cache")
OVERLAP = 0.2


def cache_path(det, source, split, mode):
    return CACHE / det / f"{source}_{split}_{mode}.pkl"


def load(det, source, split, mode):
    with open(cache_path(det, source, split, mode), "rb") as f:
        return pickle.load(f)


def run_pass(detector, img, mode):
    if mode.startswith("full"):
        if hasattr(detector, "imgsz") and len(mode) > 4:
            detector.imgsz = int(mode[4:])
        return detector.detect_raw(img, PHRASES)
    size = int(mode[1:])
    if hasattr(detector, "imgsz"):
        detector.imgsz = 640
    outs = []
    for (x0, y0, x1, y1) in slice_windows(img.shape[0], img.shape[1], size, OVERLAP):
        r = detector.detect_raw(np.ascontiguousarray(img[y0:y1, x0:x1]), PHRASES)
        if len(r):
            r.boxes = r.boxes + np.array([x0, y0, x0, y0], np.float32)
        outs.append(r)
    return RawDet.cat(outs)


def run_cache(det_name, source, split, modes, image_ids=None, data_root="data", detector=None):
    import json
    gt = json.load(open(Path(data_root) / source / split / "annotations.json"))
    imgs = gt["images"] if image_ids is None else [im for im in gt["images"] if im["id"] in set(image_ids)]
    detector = detector or build(det_name)
    for mode in modes:
        p = cache_path(det_name, source, split, mode)
        store = {"raw": {}, "sec": {}}
        if p.exists():
            with open(p, "rb") as f:
                store = pickle.load(f)
        todo = [im for im in imgs if im["id"] not in store["raw"]]
        if not todo:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        # warm-up (text encoding, lazy init) so timings are steady-state
        warm = np.array(Image.open(Path(data_root) / source / split / "images" / todo[0]["file_name"]).convert("RGB"))
        run_pass(detector, warm[:640, :640], mode)
        for k, im in enumerate(tqdm(todo, desc=f"{det_name}:{source}/{split}:{mode}", mininterval=30)):
            img = np.array(Image.open(Path(data_root) / source / split / "images" / im["file_name"]).convert("RGB"))
            t = time.perf_counter()
            r = run_pass(detector, img, mode)
            store["sec"][im["id"]] = time.perf_counter() - t
            store["raw"][im["id"]] = RawDet(r.boxes.astype(np.float32), r.ps.astype(np.float16))
            if k % 50 == 49:
                pickle.dump(store, open(p, "wb"))
        pickle.dump(store, open(p, "wb"))
    return detector


def cli_cache(cfg, argv):
    import argparse
    ap = argparse.ArgumentParser("rescueeye cache")
    ap.add_argument("--detector", required=True, choices=["yoloworld", "owlv2", "gdino"])
    ap.add_argument("--sources", nargs="+", default=["seadronessee", "visdrone", "floodnet"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", nargs="+", default=["full640"])
    ap.add_argument("--subset", default=None, help="json file {source: [image ids]} restricting the images")
    ap.add_argument("--max-images", type=int, default=None, help="first N images per source (cheap runs)")
    a = ap.parse_args(argv)
    import json
    subset = json.load(open(a.subset)) if a.subset else {}
    det = None
    for s in a.sources:
        ids = subset.get(s)
        if a.max_images:
            gt = json.load(open(Path(cfg["data_root"]) / s / a.split / "annotations.json"))
            ids = [im["id"] for im in gt["images"]][: a.max_images] if ids is None else ids[: a.max_images]
        det = run_cache(a.detector, s, a.split, a.modes, ids, cfg["data_root"], det)
