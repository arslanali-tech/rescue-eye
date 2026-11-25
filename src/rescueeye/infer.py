"""One inference entry point used by the CLI, the video tracker and the Gradio app.

Engines:
  * ``student``  - distilled YOLO11n (closed set, fast, CPU real-time on slices)
  * ``teacher``  - YOLO-World open-vocabulary detector with SAHI slicing and tuned post-processing; accepts any
                    free-text prompt (e.g. "person on roof", "cow", "tractor")
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import CAT_ID
from .detectors import Detections, YoloWorld
from .pipeline import PostCfg
from .slicing import merge, slice_windows

DEFAULT_STUDENT = "weights/rescueeye_student.pt"
DEFAULT_TEACHER_CFG = "configs/teacher.json"


class TeacherEngine:
    """Open-vocabulary engine. ``prompts`` maps a *display label* to a phrase; known RescueEye classes keep their
    tuned thresholds, any other label uses ``default_conf``."""

    def __init__(self, cfg_path=DEFAULT_TEACHER_CFG, weights="yolov8s-worldv2.pt"):
        self.det = YoloWorld(weights)
        d = json.load(open(cfg_path)) if Path(cfg_path).exists() else {}
        self.cfg = PostCfg(**{**d, "passes": tuple(d.get("passes", ("full640", "s640")))}) if d else \
            PostCfg(passes=("full640", "s640"), merge="nmm", conf={c: 0.25 for c in CAT_ID})

    def __call__(self, img, prompts: dict | None = None, default_conf=0.2, slice_size=None):
        from .detectors import PROMPT_SETS
        prompts = prompts or PROMPT_SETS[self.cfg.prompt_set]
        # expand prompt ensembles: several phrases may vote for the same label
        labels, phrases, owner = [], [], []
        for k, v in prompts.items():
            if k not in labels:
                labels.append(k)
            for ph in ([v] if isinstance(v, str) else v):
                phrases.append(ph)
                owner.append(labels.index(k))
        owner = np.array(owner)
        self.det.model.set_classes(phrases)
        self.det._phr = phrases
        outs = []
        for p in self.cfg.passes:
            if p.startswith("full"):
                self.det.imgsz = int(p[4:]) if len(p) > 4 else 640
                wins = [(0, 0, img.shape[1], img.shape[0])]
            else:
                self.det.imgsz = 640
                wins = slice_windows(img.shape[0], img.shape[1], slice_size or int(p[1:]), 0.2)
            for (x0, y0, x1, y1) in wins:
                r = self.det.model.predict(np.ascontiguousarray(img[y0:y1, x0:x1, ::-1]), imgsz=self.det.imgsz,
                                           conf=0.01, iou=0.7, verbose=False, max_det=500, device="cpu")[0]
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                outs.append(Detections(b.xyxy.numpy() + np.array([x0, y0, x0, y0], np.float32), b.conf.numpy(),
                                       owner[b.cls.numpy().astype(int)]))  # label = index into `labels`
        d = Detections.cat(outs)
        thr = np.array([self.cfg.conf.get(k, default_conf) for k in labels], np.float32)
        if len(d):
            d = d.filter(d.scores >= thr[d.labels])
        d = merge(d, self.cfg.merge, self.cfg.merge_thr)
        return d, labels


class StudentEngine:
    def __init__(self, weights=DEFAULT_STUDENT, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, img, prompts=None, conf=None, slice_size=512):
        from .students import student_detect
        d = student_detect(self.model, img, size=slice_size, conf=0.01)
        d = d.filter(d.scores >= (conf or self.conf))
        d.labels = d.labels - 1
        return d, ["person", "vehicle", "boat", "flooded_building"]


def to_rescue_dets(d: Detections, labels):
    """Map engine labels back to RescueEye category ids; unknown free-text labels are dropped from the priority
    logic but still drawn by callers."""
    keep = np.array([labels[int(l)] in CAT_ID for l in d.labels], bool) if len(d) else np.zeros(0, bool)
    dd = d.filter(keep)
    dd.labels = np.array([CAT_ID[labels[int(l)]] for l in dd.labels], np.int64)
    return dd


def build_engine(kind="student", conf=None):
    if kind == "teacher":
        return TeacherEngine()
    w = os.environ.get("RESCUEEYE_STUDENT", DEFAULT_STUDENT)
    return StudentEngine(w, conf=conf or 0.25)
