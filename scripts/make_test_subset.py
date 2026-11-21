"""Seeded test subsets (60 SeaDronesSee, 60 VisDrone images; all 79 FloodNet) used for every reported number."""
import json
import random
import shutil

for s, n in [("seadronessee", 60), ("visdrone", 60)]:
    p = f"data/{s}/test/annotations.json"
    shutil.copy(p, p.replace(".json", "_full.json"))
    g = json.load(open(p))
    keep = {im["id"] for im in random.Random(1).sample(g["images"], n)}
    g["images"] = [i for i in g["images"] if i["id"] in keep]
    g["annotations"] = [a for a in g["annotations"] if a["image_id"] in keep]
    json.dump(g, open(p, "w"))
    print(s, len(g["images"]))
