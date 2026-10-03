"""Single-image results summary (for submission): training curves, final metrics, per-class gains, accuracy vs speed."""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = pathlib.Path(__file__).resolve().parent.parent
res = root / "kaggle_output" / "results"
runs = root / "kaggle_output" / "runs" / "detect"
names = {"baseline": "YOLOv8n (baseline)", "dino-cls": "YOLOv8n + DINOv2 (CLS token)", "dino-patch": "YOLOv8n + DINOv2 (patch tokens)"}
colors = {"baseline": "#7f7f7f", "dino-cls": "#1f77b4", "dino-patch": "#d62728"}

m = pd.read_csv(res / "metrics.csv", index_col=0)
pc = pd.read_csv(res / "per_class_ap50.csv", index_col=0)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})
fig, ax = plt.subplots(2, 2, figsize=(17, 11.5))

# (a) training curves
a = ax[0, 0]
for r in names:
    d = pd.read_csv(runs / r / "results.csv")
    d.columns = d.columns.str.strip()
    a.plot(d.epoch, d["metrics/mAP50-95(B)"], marker="o", ms=4, lw=2, color=colors[r], label=names[r])
a.set_xlabel("Training epoch")
a.set_ylabel("mAP@0.5:0.95 on VOC2007 test")
a.set_title("(a) Detection accuracy during training")
a.set_xticks(range(1, 21))
a.grid(alpha=0.3)
a.legend(loc="lower right")

# (b) final metrics
a = ax[0, 1]
cols = {"mAP50": "mAP@0.5", "mAP50_95": "mAP@0.5:0.95", "precision": "Precision", "recall": "Recall"}
x = np.arange(len(cols))
w = 0.26
for i, r in enumerate(names):
    vals = m.loc[r, list(cols)].values
    bars = a.bar(x + (i - 1) * w, vals, w, color=colors[r], label=names[r])
    a.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
a.set_xticks(x, cols.values())
a.set_xlabel("Metric (VOC2007 test, 4,952 images)")
a.set_ylabel("Score (0\u20131, higher is better)")
a.set_ylim(0.55, 0.96)
a.set_title("(b) Final test metrics after 20 epochs")
a.grid(axis="y", alpha=0.3)
a.legend(loc="upper center", ncol=3, fontsize=9)

# (c) per-class gain over baseline
a = ax[1, 0]
gain = pd.DataFrame({r: (pc[r] - pc["baseline"]) * 100 for r in ["dino-cls", "dino-patch"]}).sort_values("dino-patch", ascending=False)
x = np.arange(len(gain))
for i, r in enumerate(gain.columns):
    a.bar(x + (i - 0.5) * 0.4, gain[r], 0.4, color=colors[r], label=names[r])
a.axhline(0, color="black", lw=0.8)
a.set_xticks(x, gain.index, rotation=45, ha="right")
a.set_xlabel("Pascal VOC object class")
a.set_ylabel("AP@0.5 gain over baseline (percentage points)")
a.set_title("(c) Per-class improvement from the pre-trained DINOv2 features")
a.grid(axis="y", alpha=0.3)
a.legend(loc="upper right")

# (d) accuracy vs speed
a = ax[1, 1]
for r in names:
    a.scatter(m.loc[r, "FPS"], m.loc[r, "mAP50_95"], s=260, color=colors[r], edgecolor="black", zorder=3, label=names[r])
    a.annotate(f"{m.loc[r, 'mAP50_95']:.3f} mAP\n{m.loc[r, 'FPS']:.0f} FPS, {m.loc[r, 'trainable_params_M']:.2f}M trainable\n"
               f"missed objects: {int(m.loc[r, 'FN@0.25']):,}",
               (m.loc[r, "FPS"], m.loc[r, "mAP50_95"]), textcoords="offset points",
               xytext=(14, -42) if r == "dino-cls" else (14, 4), fontsize=9.5)
a.set_xlabel("Inference speed (frames per second, batch 1, Kaggle GPU)")
a.set_ylabel("mAP@0.5:0.95 on VOC2007 test")
a.set_xlim(30, 175)
a.set_ylim(0.60, 0.70)
a.set_title("(d) Accuracy vs. speed trade-off")
a.grid(alpha=0.3)
a.legend(loc="upper right", fontsize=9.5)

fig.suptitle("Object detection on Pascal VOC (20 classes): YOLOv8n with and without a frozen pre-trained DINOv2 encoder",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = root / "report" / "figures" / "results_summary.png"
fig.savefig(out, dpi=170)
print("wrote", out)
