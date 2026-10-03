"""Explain the utility of the pre-trained DINOv2 model, independently of detection training.

1. k-NN probe: how well does a frozen global embedding predict which VOC classes are present in an image?
   DINOv2 CLS embedding (self-supervised, never saw a label) vs. YOLOv8n's own global feature (avg-pooled SPPF/P5).
2. PCA of DINOv2 patch features: unsupervised object/part segmentation.
3. Nearest-neighbour retrieval with DINOv2 CLS embeddings.

Example:
    python dino_analysis.py --n-train 4000
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths

from dino_yolo import IMAGENET_MEAN, IMAGENET_STD, load_dino, resolve_device, torch_device

OUT = Path("results")


def multi_hot(files, nc):
    y = np.zeros((len(files), nc), dtype=np.float32)
    for i, f in enumerate(files):
        lf = Path(img2label_paths([str(f)])[0])
        if lf.exists() and lf.stat().st_size:
            y[i, np.unique(np.loadtxt(lf, ndmin=2)[:, 0].astype(int))] = 1
    return y


def load_batch(files, size, normalise):
    ims = [np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255 for f in files]
    x = torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2)
    if normalise:
        x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return x


@torch.no_grad()
def embed_dino(dino, files, device, bs=64, size=224):
    out = []
    for k in range(0, len(files), bs):
        out.append(dino(pixel_values=load_batch(files[k : k + bs], size, True).to(device)).pooler_output.cpu())
    return F.normalize(torch.cat(out), dim=1)


@torch.no_grad()
def embed_yolo(net, files, device, sppf_idx, bs=64, size=512):
    out = []
    for k in range(0, len(files), bs):
        emb = net.predict(load_batch(files[k : k + bs], size, False).to(device), embed=[sppf_idx])
        out.append(torch.stack(emb).cpu())
    return F.normalize(torch.cat(out), dim=1)


def average_precision(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    if l.sum() == 0:
        return np.nan
    prec = np.cumsum(l) / (np.arange(len(l)) + 1)
    return float((prec * l).sum() / l.sum())


def knn_map(train_e, train_y, test_e, test_y, k=20):
    sim = test_e @ train_e.T
    top, idx = sim.topk(k, dim=1)
    w = (top / 0.07).exp()  # temperature-weighted vote, as in the DINO k-NN protocol
    scores = (w.unsqueeze(-1) * torch.from_numpy(train_y)[idx]).sum(1) / w.sum(1, keepdim=True)
    aps = [average_precision(scores[:, c].numpy(), test_y[:, c]) for c in range(test_y.shape[1])]
    return float(np.nanmean(aps)), aps


@torch.no_grad()
def pca_figure(dino, files, device, size=448):
    x = load_batch(files, size, True).to(device)
    tok = dino(pixel_values=x).last_hidden_state[:, 1:].cpu()  # (B, N, C)
    g = size // 14
    B, N, C = tok.shape
    border = torch.zeros(g, g, dtype=torch.bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    masks = []
    for t in tok:  # per-image 1st component splits object from background
        t = t - t.mean(0)
        fg = t @ torch.pca_lowrank(t, q=3)[2][:, 0] > 0
        masks.append(~fg if fg.reshape(g, g)[border].float().mean() > 0.5 else fg)  # background owns the border
    fg = torch.cat(masks)
    flat = tok.reshape(-1, C)
    obj = flat[fg] - flat[fg].mean(0)
    comp = obj @ torch.pca_lowrank(obj, q=3)[2]  # shared colour space across images: same part -> same colour
    comp = (comp - comp.min(0).values) / (comp.max(0).values - comp.min(0).values + 1e-6)
    rgb = torch.zeros(B * N, 3)
    rgb[fg] = comp
    rgb = rgb.reshape(B, g, g, 3).numpy()
    fig, axes = plt.subplots(2, B, figsize=(3 * B, 6))
    for i, f in enumerate(files):
        axes[0, i].imshow(Image.open(f).convert("RGB").resize((size, size)))
        axes[1, i].imshow(rgb[i], interpolation="nearest")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("input", loc="left")
    axes[1, 0].set_title("DINOv2 patch-feature PCA (no labels)", loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "dino_patch_pca.png", dpi=130)
    plt.close()


def retrieval_figure(train_files, train_e, test_files, test_e, queries, k=4):
    fig, axes = plt.subplots(len(queries), k + 1, figsize=(2.6 * (k + 1), 2.6 * len(queries)))
    for r, q in enumerate(queries):
        nn_idx = (test_e[q] @ train_e.T).topk(k).indices
        for c, f in enumerate([test_files[q]] + [train_files[i] for i in nn_idx]):
            axes[r, c].imshow(Image.open(f).convert("RGB").resize((224, 224)))
            axes[r, c].axis("off")
        axes[r, 0].set_title("query (test)" if r == 0 else "")
    for c in range(1, k + 1):
        axes[0, c].set_title(f"nearest #{c} (train)")
    plt.tight_layout()
    plt.savefig(OUT / "dino_retrieval.png", dpi=110)
    plt.close()


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="VOC.yaml")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--device", default="auto", help="'auto', '0' (CUDA GPU), 'mps' or 'cpu'")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    a.device = torch_device(resolve_device(a.device))
    OUT = Path(a.out)
    OUT.mkdir(exist_ok=True)
    random.seed(0)
    data = check_det_dataset(a.data)
    names = [data["names"][i] for i in range(data["nc"])]
    train_files = sorted(f for d in data["train"] for f in Path(d).glob("*.jpg"))
    train_files = random.sample(train_files, min(a.n_train, len(train_files)))
    val_dirs = data["val"] if isinstance(data["val"], list) else [data["val"]]
    test_files = sorted(f for d in val_dirs for f in Path(d).glob("*.jpg"))
    ytr, yte = multi_hot(train_files, data["nc"]), multi_hot(test_files, data["nc"])

    dino = load_dino().to(a.device).eval()
    yolo = YOLO("yolov8n.pt").model.float().to(a.device).eval()
    sppf_idx = next(i for i, m in enumerate(yolo.model) if type(m).__name__ == "SPPF")

    rows = []
    feats = {}
    for name, fn in [("DINOv2 ViT-S/14 CLS (self-supervised, frozen)", lambda fs: embed_dino(dino, fs, a.device)),
                     ("YOLOv8n SPPF/P5 avg-pool (COCO-supervised)", lambda fs: embed_yolo(yolo, fs, a.device, sppf_idx))]:
        tr, te = fn(train_files), fn(test_files)
        feats[name] = (tr, te)
        m, aps = knn_map(tr, ytr, te, yte)
        rows.append((name, tr.shape[1], m, aps))
        print(f"{name:48s} dim={tr.shape[1]:4d}  k-NN multi-label mAP = {m:.4f}")

    with open(OUT / "dino_knn_probe.md", "w") as f:
        f.write(f"k-NN (k=20) multi-label scene classification on VOC2007 test ({len(test_files)} images), "
                f"memory bank = {len(train_files)} random VOC train images.\n\n")
        f.write("| Global embedding | dim | mAP |\n|---|---|---|\n")
        for name, dim, m, _ in rows:
            f.write(f"| {name} | {dim} | {m:.4f} |\n")
        f.write("\nPer-class AP:\n\n| class | " + " | ".join(r[0].split(" (")[0] for r in rows) + " |\n|---|" + "---|" * len(rows) + "\n")
        for c, n in enumerate(names):
            f.write(f"| {n} | " + " | ".join(f"{r[3][c]:.3f}" for r in rows) + " |\n")

    pca_figure(dino, random.sample(test_files, 6), a.device)
    tr, te = feats[rows[0][0]]
    retrieval_figure(train_files, tr, test_files, te, random.sample(range(len(test_files)), 4))
    print("saved results/dino_knn_probe.md, results/dino_patch_pca.png, results/dino_retrieval.png")


if __name__ == "__main__":
    main()
