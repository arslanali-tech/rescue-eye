"""Teacher-student distillation and supervised baselines with a CPU-real-time YOLO11n student.

Every student is a COCO-pretrained YOLO11n fine-tuned on 512 px tiles cut from the pool images, with labels
coming from ground truth (label budget 1% / 10% / 100%), from the open-vocabulary teacher (pseudo-labels), or
both. All runs use the same tile size, augmentation and a similar number of gradient steps."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from . import cache as C
from .budget import SOURCES, load_budgets
from .detectors import Detections
from .evaluate import coco_eval
from .pipeline import PostCfg, apply_all
from .slicing import merge, slice_windows
from .tiles import build_yolo_dataset

TILE = 512
DEFAULT_TEACHER = PostCfg(prompt_set="specific", passes=("full640", "s640"), merge="nmm", merge_thr=0.5,
                          conf={"person": 0.25, "vehicle": 0.25, "boat": 0.25, "flooded_building": 0.25})


def gt_items(source, split="pool", ids=None, data_root="data"):
    gt = json.load(open(Path(data_root) / source / split / "annotations.json"))
    keep = None if ids is None else set(ids)
    by = {}
    for a in gt["annotations"]:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        by.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], a["category_id"]))
    items = []
    for im in gt["images"]:
        if keep is not None and im["id"] not in keep:
            continue
        bl = by.get(im["id"], [])
        items.append((str(Path(data_root) / source / split / "images" / im["file_name"]),
                      [b for b, _ in bl], [c for _, c in bl]))
    return items


def pseudo_items(source, cfg: PostCfg, ids=None, det="yoloworld", data_root="data"):
    gt = json.load(open(Path(data_root) / source / "pool" / "annotations.json"))
    imgs = [im for im in gt["images"] if ids is None or im["id"] in set(ids)]
    raws = {p: C.load(det, source, "pool", p)["raw"] for p in cfg.passes}
    dets = apply_all(raws, cfg, [im["id"] for im in imgs])
    return [(str(Path(data_root) / source / "pool" / "images" / im["file_name"]),
             dets[im["id"]].boxes.tolist(), dets[im["id"]].labels.tolist()) for im in imgs]


def pool_ids(source, data_root="data"):
    return [im["id"] for im in json.load(open(Path(data_root) / source / "pool" / "annotations.json"))["images"]]


def build_run_items(run: str, teacher_cfg: PostCfg | None = None):
    """Return (gt_items, pseudo_items) for a named experiment."""
    b = load_budgets()
    if run == "sup_1pct":
        return sum((gt_items(s, ids=b["1pct"][s]) for s in SOURCES), []), []
    if run == "sup_10pct":
        return sum((gt_items(s, ids=b["10pct"][s]) for s in SOURCES), []), []
    if run == "sup_100pct":
        return sum((gt_items(s) for s in SOURCES), []), []
    if run == "sup_visdrone_only":
        return gt_items("visdrone"), []
    if run == "pseudo_0pct":
        return [], sum((pseudo_items(s, DEFAULT_TEACHER) for s in SOURCES), [])
    if run == "pseudo_10pct":
        gi = sum((gt_items(s, ids=b["10pct"][s]) for s in SOURCES), [])
        pi = sum((pseudo_items(s, teacher_cfg, [i for i in pool_ids(s) if i not in set(b["10pct"][s])])
                  for s in SOURCES), [])
        return gi, pi
    raise ValueError(run)


def train_student(run, gi, pi, iters=3000, max_tiles=600, seed=0, threads=4, out_weights="weights"):
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(threads)
    # GT tiles are always kept; pseudo-labelled tiles fill the rest of the tile budget
    yg, ng = (build_yolo_dataset(f"{run}_gt", gi, max_tiles=max_tiles, seed=seed) if gi else (None, 0))
    rem = max(0, max_tiles - ng)
    yp, npl = (build_yolo_dataset(f"{run}_pl", pi, max_tiles=rem, seed=seed) if pi and rem else (None, 0))
    root = Path("runs/datasets") / run
    if root.exists():
        shutil.rmtree(root)
    for sub in ("images", "labels", "val/images", "val/labels"):
        (root / sub).mkdir(parents=True)
    for y in [yg, yp]:
        if y is None:
            continue
        src = Path(y).parent
        for sub in ("images", "labels", "val/images", "val/labels"):
            for f in (src / sub).iterdir():
                shutil.copy(f, root / sub / f.name)
        shutil.rmtree(src)
    n = len(list((root / "images").iterdir()))
    (root / "data.yaml").write_text(f"path: {root.resolve()}\ntrain: images\nval: val/images\nnames:\n"
                                    "  0: person\n  1: vehicle\n  2: boat\n  3: flooded_building\n")
    epochs = int(np.clip(round(iters / max(n, 1)), 3, 60))
    t = time.time()
    m = YOLO("yolo11n.pt")
    m.train(data=str(root / "data.yaml"), epochs=epochs, imgsz=TILE, batch=16, workers=2, device="cpu",
            project=str(Path("runs/train").resolve()), name=run, exist_ok=True, plots=False, val=False,
            seed=seed, deterministic=False, close_mosaic=max(1, epochs // 5), patience=1000, verbose=False,
            amp=False)
    dt = time.time() - t
    Path(out_weights).mkdir(exist_ok=True)
    w = Path(out_weights) / f"{run}.pt"
    shutil.copy(Path("runs/train") / run / "weights" / "last.pt", w)
    info = {"run": run, "gt_tiles": ng, "pseudo_tiles": npl, "tiles": n, "epochs": epochs,
            "gt_images": len(gi), "pseudo_images": len(pi), "train_min": dt / 60}
    Path("outputs/results/students").mkdir(parents=True, exist_ok=True)
    json.dump(info, open(f"outputs/results/students/{run}_train.json", "w"), indent=1)
    return w, info


def student_detect(model, img, size=TILE, overlap=0.2, conf=0.01, full=True):
    """Sliced inference for the student (tiles at native resolution) + optional full-image pass, NMS merge."""
    h, w = img.shape[:2]
    outs = []
    wins = slice_windows(h, w, size, overlap)
    crops = [np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]) for (x0, y0, x1, y1) in wins]
    rs = model.predict(crops, imgsz=size, conf=conf, iou=0.6, verbose=False, max_det=300, device="cpu")
    for (x0, y0, _, _), r in zip(wins, rs):
        b = r.boxes
        if b is not None and len(b):
            outs.append(Detections(b.xyxy.numpy() + np.array([x0, y0, x0, y0], np.float32), b.conf.numpy(),
                                   b.cls.numpy().astype(int) + 1))
    if full:
        r = model.predict(img[:, :, ::-1].copy(), imgsz=1024, conf=conf, iou=0.6, verbose=False, max_det=300,
                          device="cpu")[0]
        b = r.boxes
        if b is not None and len(b):
            outs.append(Detections(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy().astype(int) + 1))
    return merge(Detections.cat(outs), "nms", 0.5)


def eval_student(weights, run, data_root="data", threads=4):
    from PIL import Image
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(threads)
    m = YOLO(str(weights))
    rows = []
    for s in SOURCES:
        gt = json.load(open(Path(data_root) / s / "test" / "annotations.json"))
        dets, secs = {}, []
        for im in gt["images"]:
            img = np.array(Image.open(Path(data_root) / s / "test" / "images" / im["file_name"]).convert("RGB"))
            t = time.perf_counter()
            dets[im["id"]] = student_detect(m, img)
            secs.append(time.perf_counter() - t)
        r = coco_eval(gt, dets, s)
        rows.append({"run": run, "source": s, **r, "sec_per_image": float(np.mean(secs))})
        print(run, s, f"AP50={r['AP50']:.3f}", flush=True)
    json.dump(rows, open(f"outputs/results/students/{run}_eval.json", "w"), indent=1)
    return rows


def cli_students(cfg, argv):
    import argparse
    ap = argparse.ArgumentParser("rescueeye students")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("RESCUEEYE_THREADS", 4)))
    ap.add_argument("--teacher", default="outputs/results/saea/teacher_best.json")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)
    for run in a.runs:
        if not a.eval_only:
            tcfg = None
            if run == "pseudo_10pct":
                d = json.load(open(a.teacher))
                d["passes"] = tuple(d["passes"])
                tcfg = PostCfg(**d)
            gi, pi = build_run_items(run, tcfg)
            w, info = train_student(run, gi, pi, iters=a.iters, threads=a.threads)
            print(info, flush=True)
        eval_student(Path("weights") / f"{run}.pt", run, threads=a.threads)
