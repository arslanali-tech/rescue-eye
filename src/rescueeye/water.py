"""Light-weight flood-water segmentation for UAV RGB images (no GPU): a gradient-boosted per-pixel classifier on
colour and local-texture features, trained on FloodNet masks (water + flooded road = flood water).

It is used only to give detections context (is this person in or next to water?). A FloodWatch flood mask (or
any binary mask) can be supplied instead."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np

WATER_IDS = (3, 5)  # FloodNet: road-flooded, water
WORK_SIDE = 480


def features(img_rgb: np.ndarray, side=WORK_SIDE):
    h, w = img_rgb.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img_rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32) / np.array([180, 255, 255], np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    L, S = lab[..., 0], hsv[..., 1]
    feats = [f, hsv, lab]
    for k in (7, 21, 41):
        for ch in (L, S):
            m = cv2.blur(ch, (k, k))
            sd = np.sqrt(np.clip(cv2.blur(ch * ch, (k, k)) - m * m, 0, None))
            feats += [m[..., None], sd[..., None]]
    gx, gy = cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1)
    feats.append(cv2.blur(np.sqrt(gx * gx + gy * gy), (5, 5))[..., None])
    F = np.concatenate(feats, -1)
    return F, small.shape[:2]


class WaterSegmenter:
    def __init__(self, model=None):
        self.model = model

    def fit(self, images, masks, n_per_image=8000, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        rng = np.random.default_rng(seed)
        X, y = [], []
        for img, m in zip(images, masks):
            F, (h, w) = features(img)
            lab = cv2.resize(np.isin(m, WATER_IDS).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            idx = rng.choice(h * w, size=min(n_per_image, h * w), replace=False)
            X.append(F.reshape(-1, F.shape[-1])[idx]); y.append(lab.reshape(-1)[idx])
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=63,
                                                    random_state=seed)
        self.model.fit(np.concatenate(X), np.concatenate(y))
        return self

    def predict_proba(self, img):
        F, (h, w) = features(img)
        p = self.model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(h, w).astype(np.float32)
        p = cv2.GaussianBlur(p, (5, 5), 0)
        return cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    def predict(self, img, thr=0.5):
        return self.predict_proba(img) >= thr

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        return WaterSegmenter(joblib.load(path))


def _load_floodnet(split, data_root="data"):
    from PIL import Image
    gt = json.load(open(Path(data_root) / "floodnet" / split / "annotations.json"))
    for im in gt["images"]:
        img = np.array(Image.open(Path(data_root) / "floodnet" / split / "images" / im["file_name"]).convert("RGB"))
        m = cv2.imread(str(Path(data_root) / "floodnet" / split / "masks" / im["file_name"].replace(".jpg", ".png")), 0)
        yield im, img, m


def train_and_eval(data_root="data", out="weights/water_hgb.joblib"):
    imgs, masks = zip(*[(i, m) for _, i, m in _load_floodnet("pool", data_root)])
    seg = WaterSegmenter().fit(imgs, masks)
    Path(out).parent.mkdir(exist_ok=True)
    seg.save(out)
    inter = union = 0
    inter_fl = union_fl = 0
    for im, img, m in _load_floodnet("test", data_root):
        p = seg.predict(img)
        g = np.isin(m, WATER_IDS)
        inter += (p & g).sum(); union += (p | g).sum()
        if im["flooded_scene"]:
            inter_fl += (p & g).sum(); union_fl += (p | g).sum()
    res = {"water_IoU_all_test": float(inter / max(union, 1)), "water_IoU_flooded_scenes": float(inter_fl / max(union_fl, 1)),
           "train_images": len(imgs)}
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("outputs/results/water_segmentation.json", "w"), indent=1)
    return res


def cli_water(cfg, argv):
    print(train_and_eval(cfg["data_root"]))
