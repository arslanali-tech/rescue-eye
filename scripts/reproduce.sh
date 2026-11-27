#!/usr/bin/env bash
# Full RescueEye pipeline on CPU. Run strictly one heavy job at a time: running several PyTorch processes
# side by side on the same cores oversubscribes threads and can be ~50x slower.
set -euo pipefail
cd "$(dirname "$0")/.."
export RESCUEEYE_THREADS=${RESCUEEYE_THREADS:-$(nproc)}
R="python -m rescueeye"

$R download                                      # ~3 GB transfer, harmonised COCO json in data/
python -c "from rescueeye.budget import make_budgets; import json; r=make_budgets(); json.dump(r['10pct'], open('configs/subset_10pct.json','w'))"
python scripts/make_test_subset.py              # 60/60/79 test images (seeded)

# 1) cache raw phrase-level detections (open-vocabulary teachers)
$R cache --detector yoloworld --split test --modes full640 full1280 s640 s480 s320
$R cache --detector yoloworld --split pool --modes full640 full1280 s640 s480
$R cache --detector owlv2 --split test --modes full s960
$R cache --detector gdino --split test --modes full
$R zeroshot

# 2) surrogate-assisted tuning of the teacher on the 10% labelled subset (cheap, cached)
OMP_NUM_THREADS=1 $R tune --budget 50 --seeds 10 --out outputs/results/saea
OMP_NUM_THREADS=1 $R tune --budget 30 --seeds 20 --out outputs/results/saea_budget30
python scripts/pick_teacher.py

# 3) students
$R students --runs sup_1pct sup_10pct pseudo_0pct sup_100pct sup_visdrone_only pseudo_10pct
cp weights/pseudo_10pct.pt weights/rescueeye_student.pt

# 4) flood-water segmenter, videos, figures
$R water
bash scripts/fetch_videos.sh
$R video data/video/oleshky.webm --engine teacher --fps 5 --start 0 --duration 12 --water --name oleshky_rooftops --out outputs/video
$R video data/video/aqqala.webm --engine teacher --fps 2 --start 40 --duration 30 --name aqqala_flood --out outputs/video
$R video data/video/aqqala.webm --engine student --conf 0.35 --fps 3 --start 40 --duration 30 --water --name aqqala_flood_student --out /tmp/rescueeye_video
$R figures
python -m pytest -q
