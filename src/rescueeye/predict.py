"""`python -m rescueeye predict` and `python -m rescueeye video`: run RescueEye on your own aerial images/videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .infer import build_engine, to_rescue_dets
from .rescue import assess, heatmap, render, write_reports

KNOWN = {"person", "vehicle", "boat", "flooded_building"}


def parse_prompt(text: str | None):
    """'person, boat, flooded_building=house in water, cow' -> {label: phrase}. Unknown labels are free-text."""
    if not text:
        return None
    out = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" in part:
            k, v = [x.strip() for x in part.split("=", 1)]
        else:
            k = v = part
        out[k.replace(" ", "_") if k.replace(" ", "_") in KNOWN else k] = v
    return out


def load_water(spec, img, seg=None):
    if spec in (None, "none"):
        return None
    if spec == "auto":
        from .water import WaterSegmenter
        seg = seg or WaterSegmenter.load("weights/water_hgb.joblib")
        return seg.predict(img)
    m = np.array(Image.open(spec).convert("L").resize((img.shape[1], img.shape[0]), Image.NEAREST))
    return m > 127  # e.g. a FloodWatch flood mask exported as PNG


def run_image(path, engine, prompts=None, water="auto", geo=None, out_dir="outputs/predict", seg=None):
    img = np.array(Image.open(path).convert("RGB"))
    d, labels = engine(img, prompts) if prompts else engine(img)
    rd = to_rescue_dets(d, labels)
    wm = load_water(water, img, seg)
    findings = assess(rd, wm)
    hm = heatmap(img.shape, findings)
    stem = Path(path).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_vis = render(img, findings, water=wm, show_levels=True)
    # free-text classes that are not part of the rescue taxonomy are still drawn
    other = [(b, s, labels[int(l)]) for b, s, l in zip(d.boxes, d.scores, d.labels) if labels[int(l)] not in KNOWN]
    for b, s, name in other:
        x0, y0, x1, y1 = map(int, b)
        cv2.rectangle(det_vis, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(det_vis, f"{name} {s:.2f}", (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    Image.fromarray(det_vis).save(out / f"{stem}_detections.jpg", quality=90)
    Image.fromarray(render(img, findings, water=None, hm=hm)).save(out / f"{stem}_priority.jpg", quality=90)
    geod = None
    if geo:
        lat, lon, gsd = [float(x) for x in geo.split(",")]
        geod = {"lat": lat, "lon": lon, "gsd_m": gsd, "width": img.shape[1], "height": img.shape[0]}
    s = write_reports(findings, out, stem, img.shape, geod)
    s["other_free_text_detections"] = {n: sum(1 for *_, m in other if m == n) for n in {m for *_, m in other}}
    return s, det_vis, hm, findings


def cli_predict(cfg, argv):
    ap = argparse.ArgumentParser("rescueeye predict")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--engine", default="student", choices=["student", "teacher"])
    ap.add_argument("--prompt", default=None, help="teacher only: comma-separated classes/phrases")
    ap.add_argument("--water", default="auto", help="auto | none | path to a binary mask PNG (e.g. FloodWatch)")
    ap.add_argument("--geo", default=None, help="lat,lon,gsd_m of the image centre (north-up) for GeoJSON")
    ap.add_argument("--out", default="outputs/predict")
    a = ap.parse_args(argv)
    eng = build_engine(a.engine)
    for p in a.images:
        s, *_ = run_image(p, eng, parse_prompt(a.prompt), a.water, a.geo, a.out)
        print(p, s)


def cli_video(cfg, argv):
    from .video import process_video
    from .water import WaterSegmenter
    ap = argparse.ArgumentParser("rescueeye video")
    ap.add_argument("video")
    ap.add_argument("--engine", default="student", choices=["student", "teacher"])
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--water", action="store_true", help="also segment water for in-water priority")
    ap.add_argument("--out", default="outputs/video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--conf", type=float, default=None, help="student score threshold (default 0.25)")
    a = ap.parse_args(argv)
    seg = WaterSegmenter.load("weights/water_hgb.joblib") if a.water else None
    s = process_video(a.video, build_engine(a.engine, a.conf), a.out, a.name or Path(a.video).stem, a.fps, a.start,
                      a.duration, seg)
    print(s)
