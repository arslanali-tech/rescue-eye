"""COCO-style evaluation restricted to the classes each source annotates, plus hard-label precision/recall/F1
(used to score pseudo-labels)."""
from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from . import CAT_ID, ID_CAT
from .data import EVAL_CLASSES


def coco_from_dict(d):
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def dets_to_coco(dets: dict, cat_ids):
    """dets: {image_id: Detections} -> list of COCO result dicts (restricted to cat_ids)."""
    res = []
    for iid, d in dets.items():
        for b, s, l in zip(d.boxes, d.scores, d.labels):
            if int(l) not in cat_ids:
                continue
            res.append({"image_id": int(iid), "category_id": int(l),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s)})
    return res


def coco_eval(gt: dict, dets: dict, source: str, max_dets: int = 500):
    """Return a metrics dict: AP50-95, AP50, AP50 per class, AP50 small, recall@0.5 small/all."""
    cat_ids = [CAT_ID[c] for c in EVAL_CLASSES[source]]
    cg = coco_from_dict(gt)
    res = dets_to_coco(dets, cat_ids)
    out = {"n_images": len(gt["images"]), "n_gt": int(sum(1 for a in gt["annotations"]
                                                          if a["category_id"] in cat_ids and not a["iscrowd"]))}
    if not res:
        out.update({"AP50_95": 0.0, "AP50": 0.0, "AP30": 0.0, "AP50_small": 0.0, "R50_small": 0.0, "R50": 0.0})
        for c in cat_ids:
            out[f"AP50_{ID_CAT[c]}"] = 0.0
            out[f"AP30_{ID_CAT[c]}"] = 0.0
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes(res)
        E = COCOeval(cg, cd, "bbox")
        E.params.catIds = cat_ids
        E.params.imgIds = [im["id"] for im in gt["images"]]
        E.params.maxDets = [10, 100, max_dets]
        # IoU 0.30 is added in front of the COCO thresholds: FloodNet's mask-derived boxes are spatially offset
        # (see the data audit), so a looser match is reported alongside the standard ones.
        E.params.iouThrs = np.concatenate([[0.3], np.linspace(0.5, 0.95, 10)])
        E.evaluate()
        E.accumulate()
    P = E.eval["precision"]  # T,R,K,A,M
    R = E.eval["recall"]     # T,K,A,M

    def m(x):
        x = x[x > -1]
        return float(np.mean(x)) if x.size else float("nan")

    out["AP50_95"] = m(P[1:, :, :, 0, -1])
    out["AP50"] = m(P[1, :, :, 0, -1])
    out["AP30"] = m(P[0, :, :, 0, -1])
    out["AP50_small"] = m(P[1, :, :, 1, -1])
    out["R50_small"] = m(R[1, :, 1, -1])
    out["R50"] = m(R[1, :, 0, -1])
    for k, c in enumerate(cat_ids):
        out[f"AP50_{ID_CAT[c]}"] = m(P[1, :, k, 0, -1])
        out[f"AP30_{ID_CAT[c]}"] = m(P[0, :, k, 0, -1])
    return out


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).prod(-1)
    ab = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def gt_boxes(gt: dict):
    """{image_id: (boxes_xyxy, labels, ignore)}"""
    by = {im["id"]: ([], [], []) for im in gt["images"]}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        by[a["image_id"]][0].append([x, y, x + w, y + h])
        by[a["image_id"]][1].append(a["category_id"])
        by[a["image_id"]][2].append(bool(a["iscrowd"]))
    return {k: (np.array(v[0], np.float32).reshape(-1, 4), np.array(v[1], int), np.array(v[2], bool))
            for k, v in by.items()}


def hard_prf(gt: dict, dets: dict, source: str, iou_thr: float = 0.5):
    """Greedy one-to-one matching of hard (already thresholded) detections. Returns per-class and mean F1."""
    cat_ids = [CAT_ID[c] for c in EVAL_CLASSES[source]]
    G = gt_boxes(gt)
    stats = {c: [0, 0, 0] for c in cat_ids}  # tp, fp, fn
    for iid, (gb, gl, gi) in G.items():
        d = dets.get(iid)
        for c in cat_ids:
            g = gb[(gl == c) & ~gi]
            ig = gb[(gl == c) & gi]
            if d is None or len(d) == 0:
                stats[c][2] += len(g)
                continue
            m = d.labels == c
            pb, psc = d.boxes[m], d.scores[m]
            order = np.argsort(-psc)
            pb = pb[order]
            iou = _iou_matrix(pb, g)
            used = np.zeros(len(g), bool)
            for i in range(len(pb)):
                j = -1
                if len(g):
                    cand = np.where(~used & (iou[i] >= iou_thr))[0]
                    if len(cand):
                        j = cand[np.argmax(iou[i, cand])]
                if j >= 0:
                    used[j] = True
                    stats[c][0] += 1
                elif len(ig) and _iou_matrix(pb[i:i + 1], ig).max() >= iou_thr:
                    continue  # matched an ignore region
                else:
                    stats[c][1] += 1
            stats[c][2] += int((~used).sum())
    out = {}
    f1s = []
    for c, (tp, fp, fn) in stats.items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[ID_CAT[c]] = {"P": p, "R": r, "F1": f, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f)
    out["mean_F1"] = float(np.mean(f1s))
    return out
