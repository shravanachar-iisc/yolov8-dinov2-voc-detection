"""Train the baseline YOLOv8n or the YOLOv8n + DINOv2 variants on Pascal VOC.

Examples:
    python train.py --variant baseline
    python train.py --variant dino-cls --device 0 --cache ram      # Kaggle / CUDA
    python train.py --variant dino-cls --resume                    # continue from runs/detect/dino-cls/weights/last.pt
    python train.py --variant dino-patch --fraction 0.1 --name dino-patch-10pct
"""

import argparse
import os

from ultralytics import YOLO
from ultralytics.utils import RUNS_DIR

from dino_yolo import DinoDetectionTrainer, dino_freeze_spec, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "dino-cls", "dino-patch"], required=True)
    p.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 initialisation (COCO pre-trained)")
    p.add_argument("--data", default="VOC.yaml")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set to use")
    p.add_argument("--dino-imgsz", type=int, default=224)
    p.add_argument("--device", default="auto", help="'auto', '0' (CUDA GPU), 'mps' or 'cpu'")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--cache", choices=["none", "ram", "disk"], default="none", help="cache decoded images")
    p.add_argument("--resume", action="store_true", help="resume an interrupted run from its last.pt")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    device = resolve_device(a.device)
    if "," in device and a.variant != "baseline":
        raise SystemExit("Use a single GPU: multi-GPU DDP subprocesses do not inherit the DINO trainer settings.")
    name = a.name or a.variant
    dino_trainer = None
    if a.variant != "baseline":
        DinoDetectionTrainer.dino_mode = a.variant.split("-")[1]
        DinoDetectionTrainer.dino_imgsz = a.dino_imgsz
        dino_trainer = DinoDetectionTrainer

    if a.resume:
        last = RUNS_DIR / "detect" / name / "weights" / "last.pt"
        YOLO(str(last)).train(resume=True, trainer=dino_trainer, device=device, workers=a.workers)
        return

    model = YOLO(a.weights)
    kwargs = dict(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        fraction=a.fraction,
        device=device,
        workers=a.workers,
        cache=False if a.cache == "none" else a.cache,
        seed=0,
        name=name,
        exist_ok=True,
        plots=True,
    )
    if dino_trainer:
        kwargs.update(trainer=dino_trainer, freeze=dino_freeze_spec(model.model))
    model.train(**kwargs)


if __name__ == "__main__":
    main()
