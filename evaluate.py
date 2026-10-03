"""Evaluate trained runs on the Pascal VOC 2007 test set and produce comparison tables, plots and qualitative figures.

Example:
    python evaluate.py --runs baseline dino-cls dino-patch
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import RUNS_DIR
from ultralytics.utils.metrics import box_iou

from dino_yolo import resolve_device, torch_device  # importing dino_yolo also lets custom checkpoints unpickle

OUT = Path("results")
RUNS = RUNS_DIR / "detect"


def model_stats(model, imgsz, device):
    """Parameter counts, GFLOPs and single-image latency of the raw network (no pre/post-processing)."""
    dev = torch_device(device)
    net = model.model.float().eval().to(dev)
    total = sum(p.numel() for p in net.parameters())
    frozen = sum(p.numel() for n, p in net.named_parameters() if ".dino.encoder." in n)
    x = torch.rand(1, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            net(x)
        for _ in range(10):
            net(x)
        sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev.type, lambda: None)
        sync()
        t0 = time.perf_counter()
        n = 100
        for _ in range(n):
            net(x)
        sync()
    ms = (time.perf_counter() - t0) / n * 1000
    return dict(params_M=total / 1e6, trainable_params_M=(total - frozen) / 1e6, frozen_params_M=frozen / 1e6,
                GFLOPs=fc.get_total_flops() / 1e9, latency_ms=ms, FPS=1000 / ms)


def load_gt(img_file):
    lf = Path(img2label_paths([str(img_file)])[0])
    if not lf.exists():
        return np.zeros((0,)), np.zeros((0, 4))
    a = np.loadtxt(lf, ndmin=2)
    return a[:, 0].astype(int), a[:, 1:5]  # class, normalised xywh


def match(pred_cls, pred_xyxy, gt_cls, gt_xyxy, iou_thr=0.5):
    """Greedy per-image matching (same class, IoU >= thr). Returns TP, FP, FN counts."""
    if len(pred_cls) == 0:
        return 0, 0, len(gt_cls)
    if len(gt_cls) == 0:
        return 0, len(pred_cls), 0
    iou = box_iou(torch.tensor(pred_xyxy), torch.tensor(gt_xyxy)).numpy()
    iou[pred_cls[:, None] != gt_cls[None, :]] = 0
    used, tp = set(), 0
    for i in range(len(pred_cls)):  # predictions are already sorted by confidence
        j = int(iou[i].argmax())
        if iou[i, j] >= iou_thr and j not in used:
            used.add(j)
            tp += 1
    return tp, len(pred_cls) - tp, len(gt_cls) - tp


def per_image_errors(model, files, imgsz, device, conf):
    rows = []
    for k in range(0, len(files), 32):
        chunk = files[k : k + 32]
        for f, r in zip(chunk, model.predict(chunk, imgsz=imgsz, conf=conf, device=device, verbose=False)):
            h, w = r.orig_shape
            gc, gxywh = load_gt(f)
            gxyxy = np.stack([(gxywh[:, 0] - gxywh[:, 2] / 2) * w, (gxywh[:, 1] - gxywh[:, 3] / 2) * h,
                              (gxywh[:, 0] + gxywh[:, 2] / 2) * w, (gxywh[:, 1] + gxywh[:, 3] / 2) * h], 1) if len(gc) else np.zeros((0, 4))
            tp, fp, fn = match(r.boxes.cls.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy(), gc, gxyxy)
            rows.append(dict(image=str(f), tp=tp, fp=fp, fn=fn))
    return pd.DataFrame(rows)


def side_by_side(models, labels, img_file, imgsz, device, conf, gt_names):
    panels = []
    for m, lab in zip(models, labels):
        r = m.predict(str(img_file), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        im = r.plot(line_width=2, font_size=12)
        cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(im, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panels.append(im)
    gt = cv2.imread(str(img_file))
    h, w = gt.shape[:2]
    gc, gxywh = load_gt(img_file)
    for c, (x, y, bw, bh) in zip(gc, gxywh):
        p1, p2 = (int((x - bw / 2) * w), int((y - bh / 2) * h)), (int((x + bw / 2) * w), int((y + bh / 2) * h))
        cv2.rectangle(gt, p1, p2, (0, 200, 0), 2)
        cv2.putText(gt, gt_names[c], (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(gt, "Ground truth", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(gt, "Ground truth", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return np.hstack([gt] + panels)


def main():
    global OUT, RUNS
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["baseline", "dino-cls"])
    p.add_argument("--baseline", default="baseline", help="run used as reference for the qualitative comparison")
    p.add_argument("--compare", default=None, help="DINO run for the qualitative comparison (default: first non-baseline)")
    p.add_argument("--data", default="VOC.yaml")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", default="auto", help="'auto', '0' (CUDA GPU), 'mps' or 'cpu'")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold for error analysis / figures")
    p.add_argument("--n-qual", type=int, default=6)
    p.add_argument("--out", default="results")
    p.add_argument("--runs-dir", default=str(RUNS), help="directory containing <run>/weights/best.pt")
    a = p.parse_args()
    a.device = resolve_device(a.device)
    OUT, RUNS = Path(a.out), Path(a.runs_dir)
    OUT.mkdir(exist_ok=True)
    data = check_det_dataset(a.data)
    names = data["names"]
    ckpts = {r: RUNS / r / "weights" / "best.pt" for r in a.runs}
    val_dirs = data["val"] if isinstance(data["val"], list) else [data["val"]]
    files = sorted(f for d in val_dirs for f in Path(d).glob("*.jpg"))

    summary, per_class, errors = [], {}, {}
    for run, ck in ckpts.items():
        print(f"\n=== {run} ===")
        model = YOLO(str(ck))
        m = model.val(data=a.data, split="val", imgsz=a.imgsz, batch=32, device=a.device, plots=False,
                      project=str(OUT / "val"), name=run, exist_ok=True, verbose=False)
        ap50 = np.zeros(len(names))
        ap50[m.box.ap_class_index] = m.box.ap50
        per_class[run] = ap50
        row = dict(run=run, mAP50=m.box.map50, mAP50_95=m.box.map, precision=m.box.mp, recall=m.box.mr)
        row.update(model_stats(YOLO(str(ck)), a.imgsz, a.device))
        err = per_image_errors(YOLO(str(ck)), files, a.imgsz, a.device, a.conf)
        errors[run] = err
        row.update({f"TP@{a.conf}": int(err.tp.sum()), f"FP@{a.conf}": int(err.fp.sum()), f"FN@{a.conf}": int(err.fn.sum())})
        summary.append(row)
        print(row)

    df = pd.DataFrame(summary).set_index("run")
    df.to_csv(OUT / "metrics.csv", float_format="%.4f")
    pc = pd.DataFrame(per_class, index=[names[i] for i in range(len(names))])
    pc.to_csv(OUT / "per_class_ap50.csv", float_format="%.4f")
    (OUT / "metrics.md").write_text(df.to_markdown(floatfmt=".4f") + "\n\n" + pc.to_markdown(floatfmt=".4f") + "\n")
    print("\n", df.to_string(float_format=lambda v: f"{v:.4f}"))

    # per-class AP50 bars
    ax = pc.plot.bar(figsize=(14, 5), width=0.8)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-class AP@0.5 on VOC2007 test")
    ax.set_ylim(max(0, pc.values.min() - 0.1), 1)
    plt.tight_layout()
    plt.savefig(OUT / "per_class_ap50.png", dpi=150)
    plt.close()

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for run in a.runs:
        rc = pd.read_csv(RUNS / run / "results.csv")
        rc.columns = rc.columns.str.strip()
        axes[0].plot(rc.epoch, rc["metrics/mAP50(B)"], label=run)
        axes[1].plot(rc.epoch, rc["metrics/mAP50-95(B)"], label=run)
        axes[2].plot(rc.epoch, rc["val/cls_loss"], label=run)
    for ax, t in zip(axes, ["val mAP@0.5", "val mAP@0.5:0.95", "val classification loss"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=150)
    plt.close()

    # qualitative: images where the DINO model makes fewer / more errors than the baseline
    if a.baseline in errors and len(a.runs) > 1:
        other = a.compare or [r for r in a.runs if r != a.baseline][0]
        d = errors[a.baseline].merge(errors[other], on="image", suffixes=("_base", "_dino"))
        d["gain"] = (d.fp_base + d.fn_base) - (d.fp_dino + d.fn_dino)
        d.to_csv(OUT / "per_image_errors.csv", index=False)
        qdir = OUT / "qualitative"
        qdir.mkdir(exist_ok=True)
        models = [YOLO(str(ckpts[a.baseline])), YOLO(str(ckpts[other]))]
        labels = ["YOLOv8n (baseline)", f"YOLOv8n + DINOv2 ({other})"]
        better = d.sort_values("gain", ascending=False).head(a.n_qual)
        worse = d.sort_values("gain").head(max(2, a.n_qual // 3))
        for tag, sel in [("dino_better", better), ("baseline_better", worse)]:
            for k, f in enumerate(sel.image):
                cv2.imwrite(str(qdir / f"{tag}_{k}_{Path(f).stem}.jpg"),
                            side_by_side(models, labels, f, a.imgsz, a.device, a.conf, names))
        print(f"images improved by {other}: {(d.gain > 0).sum()}, worsened: {(d.gain < 0).sum()}, same: {(d.gain == 0).sum()}")

    (OUT / "summary.json").write_text(json.dumps(df.reset_index().to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
