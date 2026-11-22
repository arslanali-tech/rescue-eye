"""Post-processing of cached raw detections: prompt set -> per-class scores -> fusion of passes ->
NMS / greedy NMM -> per-class confidence thresholds -> size filter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import CAT_ID
from .detectors import Detections, owners_for, raw_to_dets
from .slicing import merge


@dataclass
class PostCfg:
    prompt_set: str = "specific"
    passes: tuple = ("full640",)
    merge: str = "nms"
    merge_thr: float = 0.5
    conf: dict = field(default_factory=lambda: {c: 0.0 for c in CAT_ID})  # hard-label thresholds per class
    pre_conf: float = 0.02   # scores below this are dropped before merging (speed)
    min_side: float = 0.0    # drop boxes whose shorter side is below this (px)
    max_det: int = 500

    def to_dict(self):
        d = asdict(self)
        d["passes"] = list(self.passes)
        return d


def apply(raws_by_pass: dict, cfg: PostCfg) -> Detections:
    """raws_by_pass: {pass_name: RawDet} for one image."""
    own = owners_for(cfg.prompt_set)
    thr = max(cfg.pre_conf, min(cfg.conf.values()) if cfg.conf else 0.0)
    d = Detections.cat([raw_to_dets(raws_by_pass[p], own, thr) for p in cfg.passes])
    if len(d) == 0:
        return d
    if cfg.conf:
        t = np.zeros(5, np.float32)
        for c, v in cfg.conf.items():
            t[CAT_ID[c]] = v
        d = d.filter(d.scores >= t[d.labels])
    if cfg.min_side > 0 and len(d):
        wh = d.boxes[:, 2:] - d.boxes[:, :2]
        d = d.filter(wh.min(1) >= cfg.min_side)
    d = merge(d, cfg.merge, cfg.merge_thr)
    if len(d) > cfg.max_det:
        d = d.filter(np.argsort(-d.scores)[: cfg.max_det])
    return d


def apply_all(cache_by_pass: dict, cfg: PostCfg, image_ids=None) -> dict:
    """cache_by_pass: {pass: {image_id: RawDet}} -> {image_id: Detections}"""
    first = cache_by_pass[cfg.passes[0]]
    ids = image_ids if image_ids is not None else list(first.keys())
    return {i: apply({p: cache_by_pass[p][i] for p in cfg.passes}, cfg) for i in ids}
