"""All README figures and tables from the saved results (no model is re-run, except for the qualitative panels)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = Path("outputs/results")
F = Path("outputs/figures")
SRC_LABEL = {"seadronessee": "SeaDronesSee (sea)", "visdrone": "VisDrone (city)", "floodnet": "FloodNet (Harvey flood)"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def zeroshot_figures():
    df = pd.read_csv(R / "zeroshot_all.csv").drop_duplicates(["detector", "passes", "merge", "prompt_set", "source"])
    base = df[(df.prompt_set == "specific")]
    # best merge per (detector, passes)
    rows = []
    for (d, p), g in base.groupby(["detector", "passes"]):
        best_merge = g.groupby("merge").AP50.mean().idxmax()
        rows.append(g[g["merge"] == best_merge])
    base = pd.concat(rows)
    order = base.groupby(["detector", "passes"]).AP50.mean().sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(order) + 1.2), sharey=True)
    for ax, s in zip(axes, ["seadronessee", "visdrone", "floodnet"]):
        g = base[base.source == s].set_index(["detector", "passes"]).loc[order]
        y = np.arange(len(order))
        ax.barh(y - 0.2, g.AP50, 0.4, label="AP50", color="#1f77b4")
        ax.barh(y + 0.2, g.R50_small.fillna(0), 0.4, label="recall@0.5, small objects", color="#ff7f0e")
        ax.set_yticks(y, [f"{d} | {p}" for d, p in order])
        ax.set_title(SRC_LABEL[s])
        ax.set_xlim(0, 1)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Zero-shot open-vocabulary detection (no training): full image vs SAHI slices")
    fig.tight_layout()
    fig.savefig(F / "zeroshot_passes.png", bbox_inches="tight")
    plt.close(fig)

    # prompt ablation heatmap
    pa = df[(df["merge"] == "nms") & (df.passes.isin(["full", "full640"]))]
    cols = [c for c in df.columns if c.startswith("AP50_") and c not in ("AP50_95", "AP50_small")]
    recs = []
    for (d, ps), g in pa.groupby(["detector", "prompt_set"]):
        r = {"detector": d, "prompt_set": ps}
        for _, row in g.iterrows():
            for c in cols:
                if not pd.isna(row.get(c, np.nan)):
                    r[f"{row.source}:{c[5:]}"] = row[c]
        recs.append(r)
    t = pd.DataFrame(recs).set_index(["detector", "prompt_set"])
    t = t[[c for c in ["seadronessee:person", "seadronessee:boat", "visdrone:person", "visdrone:vehicle",
                       "floodnet:vehicle", "floodnet:flooded_building"] if c in t.columns]]
    t.to_csv(R / "prompt_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1.5))
    im = ax.imshow(t.values, cmap="viridis", vmin=0, vmax=max(0.8, np.nanmax(t.values)))
    ax.set_xticks(range(t.shape[1]), [c.replace(":", "\n") for c in t.columns], fontsize=7)
    ax.set_yticks(range(len(t)), [f"{d} | {p}" for d, p in t.index], fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.5 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="AP50")
    ax.set_title("Prompt ablation (full-image pass): words matter")
    fig.tight_layout()
    fig.savefig(F / "prompt_ablation.png", bbox_inches="tight")
    plt.close(fig)


def saea_figure():
    tr = json.load(open(R / "saea" / "traces.json"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    cols = {"Random search": "#7f7f7f", "Genetic algorithm": "#2ca02c"}
    for m in sorted({t["method"] for t in tr}):
        Y = np.array([np.maximum.accumulate(t["y"]) for t in tr if t["method"] == m])
        mu, se = Y.mean(0), Y.std(0) / np.sqrt(len(Y))
        x = np.arange(1, Y.shape[1] + 1)
        c = cols.get(m, "#d62728")
        ax.plot(x, mu, label=f"{m} (n={len(Y)})", color=c)
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.2)
    ax.set_xlabel("evaluations of the true objective")
    ax.set_ylabel("best pseudo-label F1 so far")
    ax.set_title("Tuning the teacher on 84 labelled images")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "saea_convergence.png", bbox_inches="tight")
    plt.close(fig)


def students_figure():
    rows = []
    for p in sorted((R / "students").glob("*_eval.json")):
        rows += json.load(open(p))
    df = pd.DataFrame(rows)
    df.to_csv(R / "students_all.csv", index=False)
    order = [r for r in ["sup_1pct", "pseudo_0pct", "sup_10pct", "pseudo_10pct", "sup_100pct", "sup_visdrone_only"]
             if r in set(df.run)]
    names = {"sup_1pct": "supervised, 1% labels (9 img)", "pseudo_0pct": "distilled, 0 labels",
             "sup_10pct": "supervised, 10% labels (84 img)", "pseudo_10pct": "distilled + 10% labels",
             "sup_100pct": "supervised, 100% labels (840 img)", "sup_visdrone_only": "supervised on VisDrone only"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharey=True)
    for ax, s in zip(axes, ["seadronessee", "visdrone", "floodnet"]):
        g = df[df.source == s].set_index("run").loc[order]
        ax.barh(range(len(order)), g.AP50, color=["#9467bd" if "pseudo" in r else "#1f77b4" for r in order])
        for i, v in enumerate(g.AP50):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
        ax.set_yticks(range(len(order)), [names[r] for r in order])
        ax.set_title(SRC_LABEL[s])
        ax.set_xlim(0, 1)
        ax.set_xlabel("AP50 (test)")
    fig.suptitle("YOLO11n students: label budget vs open-vocabulary distillation")
    fig.tight_layout()
    fig.savefig(F / "students_label_budget.png", bbox_inches="tight")
    plt.close(fig)
    return df


def latency_figure(zs: pd.DataFrame, st: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    agg = zs[zs.prompt_set == "specific"].groupby(["detector", "passes", "merge"]).agg(
        AP50=("AP50", "mean"), sec=("sec_per_image", "mean")).reset_index()
    agg = agg.loc[agg.groupby(["detector", "passes"]).AP50.idxmax()]
    mk = {"YOLO-World-v2-S": "o", "OWLv2-B/16": "s", "Grounding-DINO-T": "^"}
    for _, r in agg.iterrows():
        col = {"YOLO-World-v2-S": "#2ca02c", "OWLv2-B/16": "#ff7f0e", "Grounding-DINO-T": "#d62728"}.get(r.detector)
        ax.scatter(r.sec, r.AP50, marker=mk.get(r.detector, "o"), s=40, label=r.detector, color=col)
        ax.annotate(r.passes, (r.sec, r.AP50), fontsize=6, xytext=(3, 2), textcoords="offset points")
    sa = st.groupby("run").agg(AP50=("AP50", "mean"), sec=("sec_per_image", "mean")).reset_index()
    for _, r in sa.iterrows():
        ax.scatter(r.sec, r.AP50, marker="*", s=90, color="#9467bd" if "pseudo" in r.run else "#1f77b4")
        ax.annotate(r.run, (r.sec, r.AP50), fontsize=6, xytext=(3, -8), textcoords="offset points")
    h, lab = ax.get_legend_handles_labels()
    u = dict(zip(lab, h))
    ax.legend(u.values(), u.keys(), fontsize=7, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlabel("CPU seconds per image (8 threads, log scale)")
    ax.set_ylabel("mean AP50 over the 3 test sets")
    ax.set_title("Accuracy vs CPU latency (stars = YOLO11n students)")
    fig.tight_layout()
    fig.savefig(F / "accuracy_vs_latency.png", bbox_inches="tight")
    plt.close(fig)


def main():
    F.mkdir(parents=True, exist_ok=True)
    zeroshot_figures()
    saea_figure()
    st = students_figure()
    latency_figure(pd.read_csv(R / "zeroshot_all.csv"), st)


def cli_figures(cfg, argv):
    main()


def showcase(images=None, out="outputs/figures/rescue_showcase.png"):
    """Qualitative panel: input | detections coloured by priority (+ water) | priority heat-map."""
    from PIL import Image as PImage

    from .infer import build_engine
    from .predict import run_image
    from .water import WaterSegmenter
    images = images or sorted(Path("app/examples").glob("*.jpg"))
    eng = build_engine("student")
    seg = WaterSegmenter.load("weights/water_hgb.joblib")
    fig, axes = plt.subplots(len(images), 3, figsize=(13, 3.3 * len(images)))
    axes = np.atleast_2d(axes)
    for r, p in enumerate(images):
        s, det_vis, hm, findings = run_image(p, eng, None, "auto", None, "outputs/predict", seg)
        img = np.array(PImage.open(p).convert("RGB"))
        pri = np.array(PImage.open(Path("outputs/predict") / f"{Path(p).stem}_priority.jpg"))
        lv = s["levels"]
        for ax, im, t in zip(axes[r], [img, det_vis, pri],
                             ["input", f"detections: {s['counts']}",
                              f"priority: CRITICAL {lv['CRITICAL']}, HIGH {lv['HIGH']}, MEDIUM {lv['MEDIUM']}"]):
            ax.imshow(im)
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)


def teacher_vs_student(image, out="outputs/figures/teacher_vs_student.png", prompts=None):
    from PIL import Image as PImage

    from .infer import build_engine, to_rescue_dets
    from .rescue import assess, render
    img = np.array(PImage.open(image).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, kind in zip(axes, ["teacher", "student"]):
        eng = build_engine(kind)
        import time
        t = time.perf_counter()
        d, labels = eng(img)
        dt = time.perf_counter() - t
        f = assess(to_rescue_dets(d, labels), None)
        ax.imshow(render(img, f, show_levels=False))
        name = "teacher: YOLO-World + SAHI (open-vocabulary)" if kind == "teacher" else "student: distilled YOLO11n"
        ax.set_title(f"{name}  |  {len(f)} objects  |  {dt:.2f} s on CPU", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
