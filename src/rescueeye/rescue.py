"""Turn detections into a rescue-priority picture: who is in or next to flood water, who has no boat nearby,
where the hot spots are. Produces an annotated image, a priority heat-map, a CSV, a GeoJSON and a JSON summary.

Priority rule (transparent and editable, not learned):
    person in water, no boat within R           -> CRITICAL
    person in / near water (boat nearby or edge) -> HIGH
    vehicle in water, flooded building           -> HIGH
    boat (possible responder or drifting)        -> MEDIUM
    anything on dry ground                       -> LOW
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import ID_CAT
from .detectors import Detections

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
LEVEL_W = {"LOW": 0.15, "MEDIUM": 0.4, "HIGH": 0.75, "CRITICAL": 1.0}
LEVEL_COL = {"LOW": (90, 200, 90), "MEDIUM": (0, 200, 255), "HIGH": (0, 120, 255), "CRITICAL": (0, 0, 255)}  # BGR
CLASS_COL = {"person": (0, 0, 255), "vehicle": (255, 160, 0), "boat": (255, 0, 200), "flooded_building": (0, 220, 255)}


@dataclass
class Finding:
    idx: int
    cls: str
    score: float
    box: tuple
    water_frac: float
    dist_to_water_px: float
    boat_nearby: bool
    level: str


def assess(dets: Detections, water: np.ndarray | None, near_factor=1.5, boat_radius_factor=6.0):
    """water: boolean mask (H,W) or None (then everything counts as 'unknown/dry')."""
    findings = []
    if water is not None and water.any():
        dist = cv2.distanceTransform((~water).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist = None
    boats = [b for b, l in zip(dets.boxes, dets.labels) if ID_CAT[int(l)] == "boat"]
    bc = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boats]).reshape(-1, 2)
    for i, (b, s, l) in enumerate(zip(dets.boxes, dets.scores, dets.labels)):
        cls = ID_CAT[int(l)]
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if water is not None:
            H, W = water.shape
            pad = int(0.25 * max(bw, bh))
            reg = water[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
            wf = float(reg.mean()) if reg.size else 0.0
            cx, cy = min(W - 1, max(0, (x0 + x1) // 2)), min(H - 1, max(0, (y0 + y1) // 2))
            d = float(dist[cy, cx]) if dist is not None else float("inf")
        else:
            wf, d = 0.0, float("inf")
        size = max(bw, bh)
        in_water = wf >= 0.35
        near = in_water or d <= near_factor * size
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        boat_near = bool(len(bc) and (np.linalg.norm(bc - c, axis=1) <= boat_radius_factor * size).any()) \
            if cls == "person" else False
        if cls == "person":
            level = "CRITICAL" if (in_water and not boat_near) else ("HIGH" if near else "LOW")
        elif cls == "vehicle":
            level = "HIGH" if in_water else ("MEDIUM" if near else "LOW")
        elif cls == "flooded_building":
            level = "HIGH"
        else:  # boat
            level = "MEDIUM"
        findings.append(Finding(i, cls, float(s), (x0, y0, x1, y1), wf, d if np.isfinite(d) else -1.0, boat_near, level))
    return findings


def heatmap(shape, findings, sigma_frac=0.04):
    H, W = shape[:2]
    hm = np.zeros((H, W), np.float32)
    for f in findings:
        x0, y0, x1, y1 = f.box
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if 0 <= cx < W and 0 <= cy < H:
            hm[cy, cx] += LEVEL_W[f.level] * (0.5 + 0.5 * f.score)
    s = max(3, int(sigma_frac * max(H, W)))
    hm = cv2.GaussianBlur(hm, (0, 0), s)
    return hm / hm.max() if hm.max() > 0 else hm


def render(img_rgb, findings, water=None, hm=None, show_levels=True, thickness=None):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = out.shape[:2]
    t = thickness or max(3, int(round(max(H, W) / 450)))
    if water is not None:
        ov = out.copy()
        ov[water] = (0.55 * ov[water] + 0.45 * np.array([255, 120, 0])).astype(np.uint8)
        out = ov
    if hm is not None:
        cm = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = (np.clip(hm, 0, 1) * 0.55)[..., None]
        out = (out * (1 - a) + cm * a).astype(np.uint8)
    for f in sorted(findings, key=lambda f: LEVELS.index(f.level)):
        x0, y0, x1, y1 = f.box
        col = LEVEL_COL[f.level] if show_levels else CLASS_COL[f.cls]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, t)
        if f.level in ("CRITICAL", "HIGH") or not show_levels:
            lab = f"{f.cls} {f.score:.2f}" if not show_levels else f"{f.level[0]}:{f.cls}"
            fs = 0.45 * t / 2 + 0.25
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 4)), (x0 + tw + 2, y0), col, -1)
            cv2.putText(out, lab, (x0 + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def summary(findings):
    s = {"counts": {}, "levels": {lv: 0 for lv in LEVELS}}
    for f in findings:
        s["counts"][f.cls] = s["counts"].get(f.cls, 0) + 1
        s["levels"][f.level] += 1
    s["people_in_or_near_water"] = sum(1 for f in findings if f.cls == "person" and f.level in ("CRITICAL", "HIGH"))
    s["people_without_boat_nearby_in_water"] = sum(1 for f in findings if f.level == "CRITICAL")
    return s


def pixel_to_lonlat(x, y, geo):
    """geo = {lat, lon, gsd_m, width, height}: image centre at (lat, lon), north-up, metres per pixel gsd_m.
    Small-area equirectangular approximation, good to a few metres over a drone frame."""
    dx = (x - geo["width"] / 2) * geo["gsd_m"]
    dy = (geo["height"] / 2 - y) * geo["gsd_m"]
    lat = geo["lat"] + dy / 111_320.0
    lon = geo["lon"] + dx / (111_320.0 * np.cos(np.radians(geo["lat"])))
    return float(lon), float(lat)


def write_reports(findings, out_dir, stem, image_shape, geo=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{stem}_detections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "class", "score", "x0", "y0", "x1", "y1", "water_fraction", "dist_to_water_px",
                    "boat_nearby", "priority", "lon", "lat"])
        for f in findings:
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            lon, lat = pixel_to_lonlat(cx, cy, geo) if geo else ("", "")
            w.writerow([f.idx, f.cls, f"{f.score:.3f}", *f.box, f"{f.water_frac:.2f}", f"{f.dist_to_water_px:.1f}",
                        int(f.boat_nearby), f.level, lon, lat])
    feats = []
    for f in findings:
        cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
        coords = list(pixel_to_lonlat(cx, cy, geo)) if geo else [cx, cy]
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": coords},
                      "properties": {"class": f.cls, "score": round(f.score, 3), "priority": f.level,
                                     "box_px": list(f.box), "boat_nearby": f.boat_nearby}})
    gj = {"type": "FeatureCollection", "features": feats,
          "properties": {"coordinates": "lon/lat (WGS84, approx.)" if geo else "image pixels (x, y); pass --geo for lon/lat",
                         "image_size": list(image_shape[:2])}}
    json.dump(gj, open(out / f"{stem}_priority.geojson", "w"), indent=1)
    s = summary(findings)
    json.dump(s, open(out / f"{stem}_summary.json", "w"), indent=1)
    return s
