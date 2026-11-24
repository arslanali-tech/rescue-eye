"""Pick the best SAEA configuration (on the labelled tuning subset) as the deployed teacher config."""
import json

t = json.load(open("outputs/results/saea/traces.json"))
b = max((x for x in t if x["method"].startswith("SAEA")), key=lambda x: x["best"])
for p in ("outputs/results/saea/teacher_best.json", "configs/teacher.json"):
    json.dump(b["best_cfg"], open(p, "w"), indent=1)
print("teacher F1 on tuning subset:", b["best"], b["best_cfg"])
