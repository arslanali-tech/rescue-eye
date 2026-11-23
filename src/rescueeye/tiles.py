"""Cut high-resolution aerial images into 640 px training tiles (YOLO format) so tiny objects keep their
native pixel size during training, the same way they are seen by sliced inference."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .slicing import slice_windows


def tile_image(img: Image.Image, boxes: np.ndarray, labels: np.ndarray, size=640, overlap=0.2, min_vis=0.5):
    """Yield (tile_img, yolo_lines, n_objects). boxes xyxy px, labels 1-based category ids."""
    w, h = img.size
    for (x0, y0, x1, y1) in slice_windows(h, w, size, overlap):
        lines = []
        if len(boxes):
            bx = boxes.copy()
            area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            cx0, cy0 = np.clip(bx[:, 0], x0, x1), np.clip(bx[:, 1], y0, y1)
            cx1, cy1 = np.clip(bx[:, 2], x0, x1), np.clip(bx[:, 3], y0, y1)
            vis = np.clip(cx1 - cx0, 0, None) * np.clip(cy1 - cy0, 0, None) / np.maximum(area, 1e-6)
            tw, th = x1 - x0, y1 - y0
            for i in np.where(vis >= min_vis)[0]:
                bw, bh = cx1[i] - cx0[i], cy1[i] - cy0[i]
                if bw < 2 or bh < 2:
                    continue
                lines.append(f"{labels[i] - 1} {((cx0[i] + cx1[i]) / 2 - x0) / tw:.6f} "
                             f"{((cy0[i] + cy1[i]) / 2 - y0) / th:.6f} {bw / tw:.6f} {bh / th:.6f}")
        yield img.crop((x0, y0, x1, y1)), lines, (x0, y0)


def build_yolo_dataset(name, items, out_root="runs/datasets", max_tiles=None, empty_frac=0.15, seed=0,
                       class_names=("person", "vehicle", "boat", "flooded_building")):
    """items: list of (image_path, boxes_xyxy, labels). Writes a YOLO dataset and returns its yaml path."""
    rng = random.Random(seed)
    root = Path(out_root) / name
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    pos, neg = [], []
    for path, boxes, labels in items:
        img = Image.open(path).convert("RGB")
        stem = Path(path).parent.parent.name + "_" + Path(path).parent.parent.parent.name + "_" + Path(path).stem
        for t, lines, (x0, y0) in tile_image(img, np.asarray(boxes, np.float32).reshape(-1, 4),
                                              np.asarray(labels, int)):
            (pos if lines else neg).append((t, lines, f"{stem}_{x0}_{y0}"))
    rng.shuffle(pos); rng.shuffle(neg)
    n_neg = int(round(len(pos) * empty_frac / (1 - empty_frac)))
    sel = pos + neg[:n_neg]
    if max_tiles and len(sel) > max_tiles:
        rng.shuffle(sel)
        sel = sel[:max_tiles]
    for t, lines, stem in sel:
        t.save(root / "images" / f"{stem}.jpg", quality=92)
        (root / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    # tiny val split (copy of a few training tiles) - only so Ultralytics runs; we always use last.pt
    (root / "val" / "images").mkdir(parents=True)
    (root / "val" / "labels").mkdir(parents=True)
    for t, lines, stem in sel[:8]:
        shutil.copy(root / "images" / f"{stem}.jpg", root / "val" / "images")
        shutil.copy(root / "labels" / f"{stem}.txt", root / "val" / "labels")
    y = root / "data.yaml"
    y.write_text(f"path: {root.resolve()}\ntrain: images\nval: val/images\nnames:\n" +
                 "".join(f"  {i}: {c}\n" for i, c in enumerate(class_names)))
    return y, len(sel)
