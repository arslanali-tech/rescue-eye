"""Download real aerial datasets, harmonise them into one 4-class taxonomy and write COCO json.

Sources (all public; see README for licences):
  * SeaDronesSee Object Detection v2 (Varga et al., WACV 2022) - swimmers and boats from UAVs over open water.
    Mirror: huggingface.co/datasets/dronefreak/SeaDronesSee (YOLO format, CC0 per the mirror card).
  * VisDrone2019-DET (Zhu et al., TPAMI 2021) - people and vehicles from drones over Chinese cities.
    Zips from the Ultralytics assets release.
  * FloodNet Track 1 (Rahnemoonfar et al., IEEE Access 2021) - UAV images after Hurricane Harvey with
    pixel masks (building-flooded, vehicle, water, ...). Mirror: huggingface.co/datasets/torchgeo/floodnet
    (CDLA-Permissive-1.0). Only the 400 labelled images are needed; they are pulled out of the 12.8 GB zip with
    HTTP range requests, so the full archive is never downloaded.

Unified classes: person, vehicle, boat, flooded_building. Each source is only evaluated on the classes it
annotates (SeaDronesSee: person, boat; VisDrone: person, vehicle; FloodNet: vehicle, flooded_building).
"""
from __future__ import annotations

import io
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from . import CAT_ID, CLASSES

SDS_REPO = "dronefreak/SeaDronesSee"
SDS_MAP = {0: "person", 1: "boat", 2: "boat", 3: None, 4: None}  # swimmer, boat, jetski, life_saving, buoy
VIS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
# VisDrone: 0 ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle,
# 8 awning-tricycle, 9 bus, 10 motor, 11 others
VIS_MAP = {1: "person", 2: "person", 4: "vehicle", 5: "vehicle", 6: "vehicle", 9: "vehicle"}
VIS_IGNORE_AS_VEHICLE = {3, 7, 8, 10}  # two/three-wheelers: neither rewarded nor penalised
FLOODNET_ZIP = "https://huggingface.co/datasets/torchgeo/floodnet/resolve/main/floodnet.zip"
# FloodNet mask ids
FN_BUILDING_FLOODED, FN_BUILDING_DRY, FN_ROAD_FLOODED, FN_WATER, FN_VEHICLE = 1, 2, 3, 5, 7
EVAL_CLASSES = {
    "seadronessee": ["person", "boat"],
    "visdrone": ["person", "vehicle"],
    "floodnet": ["vehicle", "flooded_building"],
}


def _resize_max(img: Image.Image, max_side: int):
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    return img, s


class CocoWriter:
    def __init__(self):
        self.images, self.annotations = [], []

    def add_image(self, file_name, w, h, source, extra=None):
        iid = len(self.images) + 1
        d = {"id": iid, "file_name": file_name, "width": w, "height": h, "source": source}
        if extra:
            d.update(extra)
        self.images.append(d)
        return iid

    def add_box(self, iid, cls, x, y, w, h, ignore=False):
        if w < 1 or h < 1:
            return
        self.annotations.append({
            "id": len(self.annotations) + 1, "image_id": iid, "category_id": CAT_ID[cls],
            "bbox": [float(x), float(y), float(w), float(h)], "area": float(w * h),
            "iscrowd": int(ignore), "ignore": int(ignore),
        })

    def dump(self, path):
        cats = [{"id": CAT_ID[c], "name": c} for c in CLASSES]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"images": self.images, "annotations": self.annotations, "categories": cats}, open(path, "w"))


def _get(url, timeout=120):
    for _ in range(4):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception:  # noqa: BLE001 - simple retry
            continue
    raise RuntimeError(f"download failed: {url}")


# ----------------------------------------------------------------------------------------- SeaDronesSee
def prepare_seadronessee(root: Path, n_pool: int, n_test: int, max_side: int, seed: int = 0):
    from huggingface_hub import list_repo_tree

    files = [f.path for f in list_repo_tree(SDS_REPO, repo_type="dataset", recursive=True)
             if getattr(f, "size", None) is not None and f.path.endswith(".jpg")]
    rng = random.Random(seed)
    train = sorted(p for p in files if "/images/train/" in p)
    val = sorted(p for p in files if "/images/val/" in p)
    picks = {"pool": rng.sample(train, n_pool), "test": rng.sample(val, n_test)}
    base = f"https://huggingface.co/datasets/{SDS_REPO}/resolve/main/"
    for split, paths in picks.items():
        out = root / "seadronessee" / split
        (out / "images").mkdir(parents=True, exist_ok=True)

        def fetch(p):
            name = Path(p).name
            lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            try:
                txt = _get(base + lab).decode()
            except RuntimeError:
                txt = ""
            dst = out / "images" / name
            if not dst.exists():
                img = Image.open(io.BytesIO(_get(base + p))).convert("RGB")
                orig = img.size
                img, _ = _resize_max(img, max_side)
                img.save(dst, quality=92)
            else:
                orig = None
            return name, txt, orig

        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(fetch, paths))
        cw = CocoWriter()
        for name, txt, _ in res:
            w, h = Image.open(out / "images" / name).size
            iid = cw.add_image(name, w, h, "seadronessee")
            for line in txt.strip().splitlines():
                c, xc, yc, bw, bh = line.split()[:5]
                cls = SDS_MAP.get(int(c))
                if cls is None:
                    continue
                xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                cw.add_box(iid, cls, xc - bw / 2, yc - bh / 2, bw, bh)
        cw.dump(out / "annotations.json")
        print(f"seadronessee/{split}: {len(cw.images)} images, {len(cw.annotations)} boxes")


# ----------------------------------------------------------------------------------------- VisDrone
def prepare_visdrone(root: Path, n_pool: int, n_test: int, max_side: int, seed: int = 0):
    rng = random.Random(seed)
    for split, part, n in [("pool", "val", n_pool), ("test", "test-dev", n_test)]:
        out = root / "visdrone" / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        cache = root / "_raw" / f"VisDrone2019-DET-{part}.zip"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if not cache.exists():
            cache.write_bytes(_get(VIS_URL.format(part), timeout=1200))
        z = zipfile.ZipFile(cache)
        imgs = sorted(n_ for n_ in z.namelist() if n_.endswith(".jpg"))
        cw = CocoWriter()
        for p in sorted(rng.sample(imgs, n)):
            name = Path(p).name
            img = Image.open(io.BytesIO(z.read(p))).convert("RGB")
            img, s = _resize_max(img, max_side)
            img.save(out / "images" / name, quality=92)
            iid = cw.add_image(name, *img.size, "visdrone")
            ann = p.replace("/images/", "/annotations/").rsplit(".", 1)[0] + ".txt"
            for line in z.read(ann).decode().strip().splitlines():
                v = [int(x) for x in line.strip().strip(",").split(",")[:6]]
                x, y, w, h, score, cat = v
                box = (x * s, y * s, w * s, h * s)
                if cat in VIS_MAP:
                    cw.add_box(iid, VIS_MAP[cat], *box)
                elif cat in VIS_IGNORE_AS_VEHICLE:
                    cw.add_box(iid, "vehicle", *box, ignore=True)
                elif cat == 0:  # ignored region: ignore for both land classes
                    cw.add_box(iid, "person", *box, ignore=True)
                    cw.add_box(iid, "vehicle", *box, ignore=True)
        cw.dump(out / "annotations.json")
        print(f"visdrone/{split}: {len(cw.images)} images, {len(cw.annotations)} boxes")


# ----------------------------------------------------------------------------------------- FloodNet
def mask_to_boxes(mask: np.ndarray, cls_id: int, min_area: int, open_px: int = 0):
    """Instance boxes from a semantic mask via connected components (8-connectivity)."""
    m = (mask == cls_id).astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def _floodnet_fetch(paths, out_dir):
    """Fetch image+mask pairs from the remote zip (one RemoteZip handle per worker thread)."""
    from remotezip import RemoteZip

    with RemoteZip(FLOODNET_ZIP) as z:
        for p in paths:
            name = Path(p).name
            raw_img, raw_mask = out_dir / "raw" / name, out_dir / "raw" / name.replace(".jpg", "_lab.png")
            if raw_img.exists() and raw_mask.exists():
                continue
            mp = p.replace("/image/", "/mask/").replace(".jpg", "_lab.png")
            raw_mask.write_bytes(z.read(mp))
            raw_img.write_bytes(z.read(p))


def prepare_floodnet(root: Path, cfg: dict, max_side: int, seed: int = 0, workers: int = 8):
    from remotezip import RemoteZip

    rng = random.Random(seed)
    with RemoteZip(FLOODNET_ZIP) as z:
        names = z.namelist()
    fl = sorted(n for n in names if "/Labeled/Flooded/image/" in n and n.endswith(".jpg"))
    dry = sorted(n for n in names if "/Labeled/Non-Flooded/image/" in n and n.endswith(".jpg"))
    rng.shuffle(fl); rng.shuffle(dry)
    a, b = cfg["flooded_pool"], cfg["flooded_test"]
    c, d = cfg["nonflooded_pool"], cfg["nonflooded_test"]
    splits = {"pool": fl[:a] + dry[:c], "test": fl[a:a + b] + dry[c:c + d]}
    raw = root / "_raw" / "floodnet"
    (raw / "raw").mkdir(parents=True, exist_ok=True)
    allp = splits["pool"] + splits["test"]
    chunks = [allp[i::workers] for i in range(workers)]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda ch: _floodnet_fetch(ch, raw), chunks))
    for split, paths in splits.items():
        out = root / "floodnet" / split
        for sub in ("images", "masks"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        cw = CocoWriter()
        for p in sorted(paths):
            name = Path(p).name
            img = Image.open(raw / "raw" / name).convert("RGB")
            mask = np.array(Image.open(raw / "raw" / name.replace(".jpg", "_lab.png")))
            img, s = _resize_max(img, max_side)
            img.save(out / "images" / name, quality=92)
            mask = cv2.resize(mask, img.size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / "masks" / name.replace(".jpg", ".png")), mask)
            flooded = "/Flooded/" in p
            iid = cw.add_image(name, *img.size, "floodnet", {"flooded_scene": flooded})
            area_scale = s * s
            for (x, y, w, h) in mask_to_boxes(mask, FN_VEHICLE, int(1500 * area_scale), open_px=1):
                cw.add_box(iid, "vehicle", x, y, w, h)
            for (x, y, w, h) in mask_to_boxes(mask, FN_BUILDING_FLOODED, int(20000 * area_scale), open_px=2):
                cw.add_box(iid, "flooded_building", x, y, w, h)
        cw.dump(out / "annotations.json")
        print(f"floodnet/{split}: {len(cw.images)} images, {len(cw.annotations)} boxes")


def prepare_all(cfg: dict):
    root = Path(cfg["data_root"])
    s, ms = cfg["seed"], cfg["max_side"]
    src = cfg["sources"]
    prepare_floodnet(root, src["floodnet"], ms, s)
    prepare_visdrone(root, src["visdrone"]["pool"], src["visdrone"]["test"], ms, s)
    prepare_seadronessee(root, src["seadronessee"]["pool"], src["seadronessee"]["test"], ms, s)


def load_split(root, source, split):
    """Return (coco_dict, image_dir)."""
    d = Path(root) / source / split
    return json.load(open(d / "annotations.json")), d / "images"


def cli_download(cfg, argv):
    prepare_all(cfg)
