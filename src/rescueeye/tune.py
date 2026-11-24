"""Tune the teacher's post-processing (prompt set, passes, fusion, per-class thresholds) on the 10% labelled
subset, comparing SAEA vs a genetic algorithm vs random search under a 50-evaluation budget.

Objective: macro F1@IoU0.5 of the hard pseudo-labels over the (source, class) pairs, i.e. exactly the quality
of the labels the student will be trained on. Raw detections are cached, so one evaluation re-runs only the
post-processing; the evaluation *count* is the budget being compared."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from . import CLASSES
from . import cache as C
from .budget import SOURCES, load_budgets
from .detectors import PROMPT_SETS
from .evaluate import hard_prf
from .pipeline import PostCfg, apply_all  # noqa: F401
from .surrogate import genetic_algorithm, random_search, saea

PROMPTS = list(PROMPT_SETS)
PASS_SETS = [("full640",), ("full1280",), ("s640",), ("s480",), ("full640", "s640"), ("full640", "s480"),
             ("full1280", "s640")]
DIM = 9


def decode(u) -> PostCfg:
    u = np.clip(np.asarray(u, float), 0, 1 - 1e-9)
    pick = lambda lst, x: lst[int(x * len(lst))]
    conf = {c: round(0.02 + 0.48 * u[4 + i], 3) for i, c in enumerate(CLASSES)}
    return PostCfg(prompt_set=pick(PROMPTS, u[0]), passes=pick(PASS_SETS, u[1]), merge=pick(["nms", "nmm"], u[2]),
                   merge_thr=round(0.3 + 0.5 * u[3], 2), conf=conf, min_side=round(12 * u[8], 1))


def key(u):
    return decode(u).to_dict()


class TeacherObjective:
    def __init__(self, det="yoloworld", split="pool", subset=None, data_root="data"):
        self.det, self.split = det, split
        self.gts, self.caches = {}, {}
        passes = sorted({p for ps in PASS_SETS for p in ps})
        for s in SOURCES:
            gt = json.load(open(Path(data_root) / s / split / "annotations.json"))
            if subset is not None:
                keep = set(subset[s])
                gt = {**gt, "images": [i for i in gt["images"] if i["id"] in keep],
                      "annotations": [a for a in gt["annotations"] if a["image_id"] in keep]}
            self.gts[s] = gt
            self.caches[s] = {p: C.load(det, s, split, p)["raw"] for p in passes}
        self.n_evals = 0

    def evaluate_cfg(self, cfg: PostCfg, detail=False):
        per = {}
        f1 = []
        for s in SOURCES:
            ids = [im["id"] for im in self.gts[s]["images"]]
            dets = apply_all(self.caches[s], cfg, ids)
            r = hard_prf(self.gts[s], dets, s)
            per[s] = r
            f1 += [v["F1"] for k, v in r.items() if k != "mean_F1"]
        self.n_evals += 1
        return (float(np.mean(f1)), per) if detail else float(np.mean(f1))

    def __call__(self, u):
        return self.evaluate_cfg(decode(u))


def run(budget=50, seeds=range(8), out="outputs/results/saea"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sub = load_budgets()["10pct"]
    f = TeacherObjective(subset=sub)
    traces = []
    for seed in seeds:
        for name, fn in [("rs", lambda: random_search(f, DIM, budget, seed)),
                         ("ga", lambda: genetic_algorithm(f, DIM, budget, seed)),
                         ("saea", lambda: saea(f, DIM, budget, seed, key=key))]:
            tr = fn()
            bx, by = tr.best()
            traces.append({"method": tr.method, "seed": seed, "y": tr.y, "best": by, "best_cfg": decode(bx).to_dict(),
                           "wall_s": tr.wall_s})
            print(f"seed {seed} {tr.method:40s} best F1 {by:.4f} ({tr.wall_s:.0f}s)", flush=True)
            json.dump(traces, open(out / "traces.json", "w"))
    return traces


def stats(traces):
    by = {}
    for t in traces:
        by.setdefault(t["method"], {})[t["seed"]] = t["best"]
    names = list(by)
    res = {"final_best": {m: {"mean": float(np.mean(list(v.values()))), "std": float(np.std(list(v.values()))),
                              "n": len(v)} for m, v in by.items()}}
    sa = [m for m in names if m.startswith("SAEA")][0]
    tests = {}
    for m in names:
        if m == sa:
            continue
        seeds = sorted(set(by[sa]) & set(by[m]))
        a, b = [by[sa][s] for s in seeds], [by[m][s] for s in seeds]
        try:
            p_w = float(wilcoxon(a, b, alternative="greater").pvalue)
        except ValueError:
            p_w = float("nan")
        p_u = float(mannwhitneyu(a, b, alternative="greater").pvalue)
        tests[f"{sa} vs {m}"] = {"wilcoxon_p": p_w, "mannwhitney_p": p_u}
    res["tests"] = tests
    return res


def holdout(traces, out):
    """Re-score every method's best configuration on the held-out *test* split (never seen during tuning)."""
    f = TeacherObjective(split="test")
    from .evaluate import coco_eval
    rows = []
    for t in traces:
        d = dict(t["best_cfg"])
        d["passes"] = tuple(d["passes"])
        cfg = PostCfg(**d)
        f1 = f.evaluate_cfg(cfg)
        rows.append({"method": t["method"], "seed": t["seed"], "tune_F1": t["best"], "test_F1": f1})
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(r["test_F1"])
    summ = {m: {"test_F1_mean": float(np.mean(v)), "test_F1_std": float(np.std(v))} for m, v in by.items()}
    json.dump({"rows": rows, "summary": summ}, open(Path(out) / "holdout.json", "w"), indent=1)
    return summ


def cli_tune(cfg, argv):
    import argparse
    ap = argparse.ArgumentParser("rescueeye tune")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="outputs/results/saea")
    ap.add_argument("--holdout-only", action="store_true")
    a = ap.parse_args(argv)
    if a.holdout_only:
        tr = json.load(open(Path(a.out) / "traces.json"))
    else:
        tr = run(a.budget, range(a.seeds), a.out)
        s = stats(tr)
        json.dump(s, open(Path(a.out) / "stats.json", "w"), indent=1)
        print(json.dumps(s, indent=1))
    print(holdout(tr, a.out))
