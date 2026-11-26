import numpy as np
from PIL import Image

from rescueeye import CAT_ID
from rescueeye.data import mask_to_boxes
from rescueeye.detectors import PHRASES, PROMPT_SETS, Detections, RawDet, owners_for, raw_to_dets
from rescueeye.evaluate import coco_eval, hard_prf
from rescueeye.pipeline import PostCfg, apply
from rescueeye.predict import parse_prompt
from rescueeye.rescue import assess, heatmap, summary
from rescueeye.slicing import greedy_nmm, nms, slice_windows
from rescueeye.surrogate import genetic_algorithm, random_search, saea
from rescueeye.tiles import tile_image
from rescueeye.tune import DIM, decode


def test_slice_windows_cover_image_with_overlap():
    for h, w, s in [(1080, 1920, 640), (765, 1360, 512), (500, 400, 640)]:
        win = slice_windows(h, w, s, 0.2)
        cov = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in win:
            assert x1 - x0 <= s and y1 - y0 <= s
            cov[y0:y1, x0:x1] = True
        assert cov.all()


def test_nms_and_nmm():
    d = Detections(np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]], np.float32),
                   np.array([0.9, 0.8, 0.7, 0.6], np.float32), np.array([1, 1, 1, 2]))
    k = nms(d, 0.5)
    assert len(k) == 3  # overlapping same-class pair suppressed, other class kept
    # a box cut by a slice border: two halves merge into the enclosing box
    h = Detections(np.array([[0, 0, 10, 10], [8, 0, 20, 10]], np.float32), np.array([0.9, 0.8], np.float32),
                   np.array([1, 1]))
    m = greedy_nmm(h, 0.1)
    assert len(m) == 1 and np.allclose(m.boxes[0], [0, 0, 20, 10])


def test_mask_to_boxes():
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:30] = 7
    m[50:90, 60:70] = 7
    b = sorted(mask_to_boxes(m, 7, min_area=50))
    assert b == [(10, 10, 20, 10), (60, 50, 10, 40)]


def _gt(boxes, cats, w=200, h=200):
    return {"images": [{"id": 1, "width": w, "height": h, "file_name": "x.jpg"}],
            "annotations": [{"id": i + 1, "image_id": 1, "category_id": c, "bbox": b, "area": b[2] * b[3],
                             "iscrowd": 0} for i, (b, c) in enumerate(zip(boxes, cats))],
            "categories": [{"id": v, "name": k} for k, v in CAT_ID.items()]}


def test_metrics_perfect_and_empty():
    gt = _gt([[10, 10, 20, 20], [100, 100, 40, 30]], [1, 3])
    d = {1: Detections(np.array([[10, 10, 30, 30], [100, 100, 140, 130]], np.float32), np.array([0.9, 0.8], np.float32),
                       np.array([1, 3]))}
    m = coco_eval(gt, d, "seadronessee")
    assert abs(m["AP50"] - 1.0) < 1e-6 and abs(m["AP50_95"] - 1.0) < 1e-6
    f = hard_prf(gt, d, "seadronessee")
    assert f["mean_F1"] == 1.0
    f0 = hard_prf(gt, {1: Detections.empty()}, "seadronessee")
    assert f0["mean_F1"] == 0.0


def test_prompt_sets_and_raw_to_dets():
    ps = np.zeros((2, len(PHRASES)), np.float16)
    ps[0, PHRASES.index("swimmer")] = 0.9
    ps[1, PHRASES.index("house surrounded by water")] = 0.8
    raw = RawDet(np.array([[0, 0, 5, 5], [10, 10, 50, 50]], np.float32), ps)
    d = raw_to_dets(raw, owners_for("context"))
    assert list(d.labels) == [CAT_ID["person"], CAT_ID["flooded_building"]]
    d2 = raw_to_dets(raw, owners_for("generic"))  # these phrases are unused by the generic set
    assert len(d2) == 0
    assert set(PROMPT_SETS) == {"generic", "specific", "context", "ensemble"}


def test_pipeline_thresholds():
    ps = np.zeros((2, len(PHRASES)), np.float16)
    ps[0, 0], ps[1, 0] = 0.9, 0.1
    raw = {"full640": RawDet(np.array([[0, 0, 5, 5], [20, 20, 30, 30]], np.float32), ps)}
    cfg = PostCfg(passes=("full640",), conf={"person": 0.5, "vehicle": 0.5, "boat": 0.5, "flooded_building": 0.5})
    assert len(apply(raw, cfg)) == 1


def test_rescue_logic():
    water = np.zeros((200, 200), bool)
    water[:, :100] = True
    d = Detections(np.array([[40, 40, 50, 50], [150, 150, 160, 160], [60, 100, 90, 120]], np.float32),
                   np.array([0.9, 0.9, 0.8], np.float32),
                   np.array([CAT_ID["person"], CAT_ID["person"], CAT_ID["vehicle"]]))
    f = assess(d, water)
    assert f[0].level == "CRITICAL" and f[1].level == "LOW" and f[2].level == "HIGH"
    s = summary(f)
    assert s["people_without_boat_nearby_in_water"] == 1
    # a boat right next to the swimmer downgrades CRITICAL -> HIGH
    d2 = Detections(np.vstack([d.boxes[:1], [[52, 40, 70, 52]]]).astype(np.float32), np.array([0.9, 0.9], np.float32),
                    np.array([CAT_ID["person"], CAT_ID["boat"]]))
    assert assess(d2, water)[0].level == "HIGH"
    hm = heatmap((200, 200), f)
    assert hm.max() <= 1.0 and hm[45, 45] > hm[155, 155]


def test_tiles_keep_visible_boxes():
    img = Image.new("RGB", (1000, 600))
    tiles = list(tile_image(img, np.array([[10, 10, 30, 30], [600, 400, 620, 420]], np.float32), np.array([1, 3]),
                            size=512))
    n_lines = sum(len(lines) for _, lines, _ in tiles)
    assert n_lines >= 2
    for _, lines, _ in tiles:
        for ln in lines:
            c, *v = ln.split()
            assert c in {"0", "2"} and all(0 <= float(x) <= 1 for x in v)


def test_optimisers_on_toy_problem():
    f = lambda u: -float(np.sum((np.asarray(u) - 0.3) ** 2))
    for fn in (random_search, genetic_algorithm):
        tr = fn(f, 3, 20, 0)
        assert len(tr.y) == 20
    tr = saea(f, 3, 20, 0, n_init=8, inner_gens=5)
    assert len(tr.y) == 20 and max(tr.y) > -0.1


def test_decode_and_prompt_parsing():
    c = decode(np.full(DIM, 0.5))
    assert c.prompt_set in PROMPT_SETS and len(c.conf) == 4
    p = parse_prompt("person, flooded building=house in water, cow")
    assert p == {"person": "person", "flooded_building": "house in water", "cow": "cow"}
