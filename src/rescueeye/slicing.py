"""Slicing-aided inference (SAHI, Akyon et al., ICIP 2022), re-implemented so that raw per-slice detections
can be cached and the merge step re-tuned without re-running the detector."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms

from .detectors import Detections


def slice_windows(h: int, w: int, size: int, overlap: float):
    """Sliding windows covering the image, evenly spread so that neighbours overlap by at least `overlap`."""
    step = max(1, int(round(size * (1 - overlap))))

    def starts(n):
        if n <= size:
            return [0]
        k = int(np.ceil((n - size) / step)) + 1  # windows needed to keep at least `overlap`
        return sorted(set(int(round(v)) for v in np.linspace(0, n - size, k)))

    return [(x, y, min(x + size, w), min(y + size, h)) for y in starts(h) for x in starts(w)]


def sliced_detect(detect_fn, img: np.ndarray, size: int, overlap: float):
    """Run ``detect_fn(crop)`` on every window and shift the boxes back to image coordinates.
    Returns the *unmerged* detections plus the per-slice extra outputs (e.g. phrase scores)."""
    h, w = img.shape[:2]
    outs = []
    for (x0, y0, x1, y1) in slice_windows(h, w, size, overlap):
        d = detect_fn(np.ascontiguousarray(img[y0:y1, x0:x1]))
        if len(d):
            d.boxes = d.boxes + np.array([x0, y0, x0, y0], np.float32)
        outs.append(d)
    return outs


def nms(d: Detections, iou: float = 0.5, class_agnostic: bool = False) -> Detections:
    if len(d) == 0:
        return d
    b = torch.as_tensor(d.boxes, dtype=torch.float32)
    s = torch.as_tensor(d.scores, dtype=torch.float32)
    lab = torch.zeros(len(d), dtype=torch.long) if class_agnostic else torch.as_tensor(d.labels.astype(np.int64))
    keep = batched_nms(b, s, lab, iou).numpy()
    return d.filter(keep)


def _ios(a: np.ndarray, b: np.ndarray):
    """Intersection over the smaller box area (numpy)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-6)


def _iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def greedy_nmm(d: Detections, thr: float = 0.5, metric: str = "ios") -> Detections:
    """Greedy non-maximum *merging* (SAHI's GREEDYNMM): overlapping same-class boxes are fused into their
    enclosing box instead of being suppressed, which repairs objects cut by slice borders."""
    if len(d) == 0:
        return d
    out_b, out_s, out_l = [], [], []
    for c in np.unique(d.labels):
        m = d.labels == c
        order = np.argsort(-d.scores[m])
        b, s = d.boxes[m][order].astype(np.float32), d.scores[m][order]
        sim = _ios(b, b) if metric == "ios" else _iou(b, b)
        alive = np.ones(len(b), bool)
        for i in range(len(b)):
            if not alive[i]:
                continue
            grp = alive & (sim[i] >= thr)
            grp[i] = True
            bb = b[grp]
            out_b.append(np.concatenate([bb[:, :2].min(0), bb[:, 2:].max(0)]))
            out_s.append(float(s[i]))
            out_l.append(int(c))
            alive &= ~grp
    return Detections(np.array(out_b, np.float32), np.array(out_s, np.float32), np.array(out_l, np.int64))


def merge(d: Detections, method: str = "nms", thr: float = 0.5) -> Detections:
    if method == "nms":
        return nms(d, thr)
    if method == "nmm":
        return greedy_nmm(d, thr, "ios")
    raise ValueError(method)
