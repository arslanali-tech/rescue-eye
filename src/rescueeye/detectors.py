"""Uniform wrappers around open-vocabulary and closed-set detectors (CPU).

Every detector exposes ``detect(img_rgb, prompts) -> Detections`` where ``prompts`` maps each RescueEye class
to one or more text phrases. Phrases are scored independently and folded back onto their class, so prompt
ensembles ("person", "swimmer", ...) are supported by every open-vocabulary backend.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import CAT_ID

torch.set_num_threads(int(os.environ.get("RESCUEEYE_THREADS", os.cpu_count() or 4)))


@dataclass
class Detections:
    boxes: np.ndarray   # (N,4) xyxy, pixels
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) RescueEye category ids (1-based)

    @staticmethod
    def empty():
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self):
        return len(self.scores)

    def filter(self, keep):
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])

    @staticmethod
    def cat(items):
        items = [d for d in items if len(d)]
        if not items:
            return Detections.empty()
        return Detections(np.concatenate([d.boxes for d in items]), np.concatenate([d.scores for d in items]),
                          np.concatenate([d.labels for d in items]))

    def to_dict(self):
        return {"boxes": self.boxes.astype(np.float32), "scores": self.scores.astype(np.float32),
                "labels": self.labels.astype(np.int16)}


def flatten_prompts(prompts: dict):
    phrases, owners = [], []
    for cls, p in prompts.items():
        for ph in ([p] if isinstance(p, str) else p):
            phrases.append(ph)
            owners.append(CAT_ID[cls])
    return phrases, np.array(owners)


# ------------------------------------------------------------------ phrase-level raw outputs (cacheable)
PHRASES = [
    "person", "swimmer", "person in water", "people",          # person
    "car", "vehicle", "truck",                                  # vehicle
    "boat", "ship", "jet ski",                                  # boat
    "building", "flooded house", "house surrounded by water", "roof",  # flooded_building
]
PHRASE_CLASS = ["person"] * 4 + ["vehicle"] * 3 + ["boat"] * 3 + ["flooded_building"] * 4
PROMPT_SETS = {
    "generic": {"person": "person", "vehicle": "vehicle", "boat": "boat", "flooded_building": "building"},
    "specific": {"person": "person", "vehicle": "car", "boat": "boat", "flooded_building": "flooded house"},
    "context": {"person": "swimmer", "vehicle": "car", "boat": "boat",
                "flooded_building": "house surrounded by water"},
    "ensemble": {c: [p for p, pc in zip(PHRASES, PHRASE_CLASS) if pc == c]
                 for c in ["person", "vehicle", "boat", "flooded_building"]},
}


@dataclass
class RawDet:
    boxes: np.ndarray   # (N,4) xyxy
    ps: np.ndarray      # (N,P) per-phrase scores (0 where not scored)

    @staticmethod
    def empty(P=len(PHRASES)):
        return RawDet(np.zeros((0, 4), np.float32), np.zeros((0, P), np.float16))

    def __len__(self):
        return len(self.boxes)

    @staticmethod
    def cat(items):
        items = [r for r in items if len(r)]
        if not items:
            return RawDet.empty()
        return RawDet(np.concatenate([r.boxes for r in items]), np.concatenate([r.ps for r in items]))


def owners_for(prompt_set):
    """Phrase -> category id (0 = phrase unused) for a prompt set name or dict."""
    ps = PROMPT_SETS[prompt_set] if isinstance(prompt_set, str) else prompt_set
    own = np.zeros(len(PHRASES), np.int64)
    for cls, v in ps.items():
        for ph in ([v] if isinstance(v, str) else v):
            own[PHRASES.index(ph)] = CAT_ID[cls]
    return own


def raw_to_dets(raw: RawDet, owners: np.ndarray, min_conf: float = 0.0) -> Detections:
    if len(raw) == 0:
        return Detections.empty()
    ps = raw.ps.astype(np.float32)
    cls_scores = np.zeros((len(raw), 5), np.float32)
    for c in range(1, 5):
        cols = owners == c
        if cols.any():
            cls_scores[:, c] = ps[:, cols].max(1)
    lab = cls_scores.argmax(1)
    sc = cls_scores.max(1)
    keep = (sc > min_conf) & (lab > 0)
    return Detections(raw.boxes[keep].astype(np.float32), sc[keep], lab[keep])


def _enable_multilabel_nms():
    """YOLO-World scores every text phrase independently (region-text contrastive head), but Ultralytics' default
    NMS keeps only the arg-max phrase per box, which makes phrases compete. Multi-label NMS keeps every
    (box, phrase) pair above the threshold so per-phrase scores stay independent and prompt sets can be compared."""
    import functools
    from ultralytics.models.yolo.detect import predict as P
    from ultralytics.utils import nms as N
    if getattr(P.nms, "_rescueeye_ml", False):
        return

    class _NS:
        pass
    ns = _NS()
    for k in dir(N):
        if not k.startswith("__"):
            setattr(ns, k, getattr(N, k))
    ns.non_max_suppression = functools.partial(N.non_max_suppression, multi_label=True)
    ns._rescueeye_ml = True
    P.nms = ns


class YoloWorld:
    name = "YOLO-World-v2-S"

    def __init__(self, weights="yolov8s-worldv2.pt", imgsz=640, conf=0.01):
        from ultralytics import YOLOWorld
        _enable_multilabel_nms()
        self.model = YOLOWorld(weights)
        self.imgsz, self.conf = imgsz, conf
        self._phr = None

    def detect_raw(self, img, phrases=PHRASES):
        if list(phrases) != self._phr:
            self.model.set_classes(list(phrases))
            self._phr = list(phrases)
        r = self.model.predict(img[:, :, ::-1], imgsz=self.imgsz, conf=self.conf, iou=0.7, verbose=False,
                               max_det=500, device="cpu")[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return RawDet.empty(len(phrases))
        ps = np.zeros((len(b), len(phrases)), np.float16)
        ps[np.arange(len(b)), b.cls.numpy().astype(int)] = b.conf.numpy()
        return RawDet(b.xyxy.numpy().astype(np.float32), ps)

    def detect(self, img, prompts):
        phrases, owners = flatten_prompts(prompts)
        if phrases != self._phr:
            self.model.set_classes(phrases)
            self._phr = phrases
        r = self.model.predict(img[:, :, ::-1], imgsz=self.imgsz, conf=self.conf, iou=0.6, verbose=False,
                               max_det=300, device="cpu")[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections.empty()
        return Detections(b.xyxy.numpy(), b.conf.numpy(), owners[b.cls.numpy().astype(int)])


class OwlV2:
    name = "OWLv2-B/16"

    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", conf=0.03):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval()
        self.conf = conf
        self._phr, self._text = None, None

    @torch.inference_mode()
    def detect_raw(self, img, phrases=PHRASES):
        h, w = img.shape[:2]
        inputs = self.proc(text=[list(phrases)], images=img, return_tensors="pt")
        out = self.model(**inputs)
        s = max(h, w)
        logits = out.logits[0].sigmoid()
        keep = logits.max(-1).values > self.conf
        if keep.sum() == 0:
            return RawDet.empty(len(phrases))
        cx, cy, bw, bh = out.pred_boxes[0][keep].unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * s
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
        return RawDet(xyxy.numpy().astype(np.float32), logits[keep].numpy().astype(np.float16))

    @torch.inference_mode()
    def detect(self, img, prompts):
        phrases, owners = flatten_prompts(prompts)
        h, w = img.shape[:2]
        inputs = self.proc(text=[phrases], images=img, return_tensors="pt")
        out = self.model(**inputs)
        # OWLv2 pads to a square; boxes are relative to the padded square of side max(h, w)
        s = max(h, w)
        logits = out.logits[0].sigmoid()            # (Q, P)
        scores, idx = logits.max(-1)
        keep = scores > self.conf
        if keep.sum() == 0:
            return Detections.empty()
        bx = out.pred_boxes[0][keep]                # cx, cy, w, h relative to padded square
        cx, cy, bw, bh = bx.unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * s
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
        return Detections(xyxy.numpy(), scores[keep].numpy(), owners[idx[keep].numpy()])


class GroundingDino:
    name = "Grounding-DINO-T"

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", conf=0.1, max_side=1333):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval()
        self.conf, self.max_side = conf, max_side

    @torch.inference_mode()
    def detect_raw(self, img, phrases=PHRASES):
        from PIL import Image
        h, w = img.shape[:2]
        text = ". ".join(p.lower() for p in phrases) + "."
        inputs = self.proc(images=Image.fromarray(img), text=text, return_tensors="pt")
        out = self.model(**inputs)
        ph = self._phrase_scores(out, inputs, len(phrases))
        keep = ph.max(-1).values > self.conf
        if keep.sum() == 0:
            return RawDet.empty(len(phrases))
        cx, cy, bw, bh = out.pred_boxes[0][keep].unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
        return RawDet(xyxy.numpy().astype(np.float32), ph[keep].numpy().astype(np.float16))

    def _phrase_scores(self, out, inputs, n_phrases):
        """Score of each phrase for each query = max sigmoid over that phrase's tokens."""
        probs = out.logits[0].sigmoid()
        ids = inputs["input_ids"][0].tolist()
        tok = self.proc.tokenizer
        dot = tok.convert_tokens_to_ids(".")
        spans, cur = [], []
        for i, t in enumerate(ids):
            if t in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
                continue
            if t == dot:
                if cur:
                    spans.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            spans.append(cur)
        assert len(spans) >= n_phrases, (len(spans), n_phrases)
        return torch.stack([probs[:, sp].max(-1).values for sp in spans[:n_phrases]], -1)

    @torch.inference_mode()
    def detect(self, img, prompts):
        from PIL import Image
        phrases, owners = flatten_prompts(prompts)
        h, w = img.shape[:2]
        pil = Image.fromarray(img)
        s = min(1.0, self.max_side / max(h, w))
        if s < 1:
            pil = pil.resize((round(w * s), round(h * s)))
        text = ". ".join(p.lower() for p in phrases) + "."
        inputs = self.proc(images=pil, text=text, return_tensors="pt")
        out = self.model(**inputs)
        # per-query score for each phrase = max token prob over the phrase's tokens
        probs = out.logits[0].sigmoid()  # (Q, 256 tokens)
        ids = inputs["input_ids"][0].tolist()
        tok = self.proc.tokenizer
        spans, cur, start = [], [], None
        dot = tok.convert_tokens_to_ids(".")
        for i, t in enumerate(ids):
            if t in (tok.cls_token_id, tok.sep_token_id):
                continue
            if t == dot:
                if cur:
                    spans.append(cur)
                cur = []
            else:
                cur.append(i)
        if cur:
            spans.append(cur)
        spans = spans[: len(phrases)]
        ph_scores = torch.stack([probs[:, sp].max(-1).values for sp in spans], -1)  # (Q, P)
        scores, idx = ph_scores.max(-1)
        keep = scores > self.conf
        if keep.sum() == 0:
            return Detections.empty()
        cx, cy, bw, bh = out.pred_boxes[0][keep].unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        return Detections(xyxy.numpy(), scores[keep].numpy(), owners[idx[keep].numpy()])


class YoloStudent:
    """Closed-set Ultralytics YOLO trained on RescueEye classes (class index i -> category id i+1)."""
    name = "YOLO11n student"

    def __init__(self, weights, imgsz=640, conf=0.02):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.conf = imgsz, conf

    def detect(self, img, prompts=None):
        r = self.model.predict(img[:, :, ::-1], imgsz=self.imgsz, conf=self.conf, iou=0.6, verbose=False,
                               max_det=300, device="cpu")[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections.empty()
        return Detections(b.xyxy.numpy(), b.conf.numpy(), b.cls.numpy().astype(int) + 1)


def build(name, **kw):
    return {"yoloworld": YoloWorld, "owlv2": OwlV2, "gdino": GroundingDino}[name](**kw)


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t
