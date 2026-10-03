#!/usr/bin/env bash
# Train all variants sequentially (identical hyper-parameters, seed and COCO-pretrained YOLOv8n initialisation).
# Extra arguments are forwarded to train.py, e.g.  ./run_all.sh --device 0 --cache ram
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs
for v in baseline dino-cls dino-patch; do
  python train.py --variant "$v" "$@" > "logs/$v.log" 2>&1
done
python evaluate.py --runs baseline dino-cls dino-patch > logs/evaluate.log 2>&1
python dino_analysis.py > logs/dino_analysis.log 2>&1
