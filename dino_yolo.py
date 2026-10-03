"""YOLOv8 + DINOv2: a frozen, self-supervised DINOv2 encoder injects global scene semantics into YOLOv8's P5 feature map.

Architecture (following https://cambum.net/I3DLab/AI4DM.htm):

    image ──► YOLOv8 CSP-Darknet backbone ──► SPPF (P5, stride 32) ──┐
      │                                                              (+) ──► YOLOv8 neck + detect head
      └──► frozen DINOv2 ViT-S/14 ──► embedding ──► 1x1 projection ──┘

The fusion `SPPF + proj(DINO)` is the same as the reference "Concat + 1x1 conv" fusion with the YOLO half of the
1x1 conv fixed to identity. `proj` is zero-initialised (ControlNet-style), so at step 0 the network is exactly the
pre-trained YOLOv8 and it learns how much DINO context to use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules.block import SPPF
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

DINO_ID = "facebook/dinov2-small"  # ViT-S/14, 22M params, 384-d embeddings, pre-trained on LVD-142M
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_device(device: str | None = None) -> str:
    """Ultralytics device string: the explicit value, else CUDA GPU '0', else Apple 'mps', else 'cpu'."""
    if device not in (None, "", "auto"):
        return str(device)
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def torch_device(device: str) -> torch.device:
    """Map an Ultralytics device string ('0', 'mps', 'cpu') to a torch.device."""
    return torch.device(f"cuda:{device}" if str(device).isdigit() else device)


def load_dino(model_id: str = DINO_ID) -> Dinov2Model:
    """Load DINOv2 from the local Hugging Face cache, downloading it on first use."""
    try:
        return Dinov2Model.from_pretrained(model_id, local_files_only=True)
    except OSError:
        return Dinov2Model.from_pretrained(model_id)


class DINOv2Context(nn.Module):
    """Frozen DINOv2 encoder that returns a projected context map of shape (B, c_out, h, w).

    mode="cls":   the global CLS embedding is broadcast over the h x w grid (the reference design).
    mode="patch": the 16x16 grid of patch embeddings is resized to h x w (spatially-aware variant).
    """

    def __init__(self, c_out: int, mode: str = "cls", model_id: str = DINO_ID, imgsz: int = 224):
        super().__init__()
        assert mode in {"cls", "patch"}, f"unknown DINO mode '{mode}'"
        assert imgsz % 14 == 0, "DINOv2 input size must be a multiple of the 14px patch size"
        self.mode, self.imgsz = mode, imgsz
        self.encoder = load_dino(model_id)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Conv2d(self.encoder.config.hidden_size, c_out, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # the foundation model always stays in inference mode
        return self

    @torch.no_grad()
    def embed(self, img: torch.Tensor) -> torch.Tensor:
        """Return DINOv2 features for a batch of RGB images in [0, 1]: (B, C, 1, 1) or (B, C, g, g)."""
        x = F.interpolate(img, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.encoder.dtype)
        out = self.encoder(pixel_values=x)
        if self.mode == "cls":
            return out.pooler_output[:, :, None, None]  # layer-normed CLS token = global image embedding
        g = self.imgsz // 14
        tokens = out.last_hidden_state[:, 1:]  # drop CLS
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, g, g)

    def forward(self, img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        feat = self.embed(img).to(self.proj.weight.dtype)
        if self.mode == "cls":
            return self.proj(feat).expand(-1, -1, *size)  # global-to-spatial broadcast
        return self.proj(F.interpolate(feat, size=size, mode="bilinear", align_corners=False))


class SPPFDino(SPPF):
    """SPPF block whose output is enriched with DINOv2 context. Keeps SPPF parameter names so YOLO weights load."""

    def __init__(self, sppf: SPPF, context: DINOv2Context):
        nn.Module.__init__(self)
        # reuse the original submodules: rebuilding them would reset YOLO's BatchNorm eps/momentum
        self.cv1, self.cv2, self.m = sppf.cv1, sppf.cv2, sppf.m
        self.n, self.add = getattr(sppf, "n", 3), getattr(sppf, "add", False)
        self.i, self.f, self.type, self.np = sppf.i, sppf.f, "SPPFDino", sppf.np
        self.dino = context
        self.img = None  # raw input image, set by YOLODinoModel for the duration of a forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y + self.dino(self.img, y.shape[-2:])


class YOLODinoModel(DetectionModel):
    """YOLOv8 DetectionModel whose SPPF (P5) layer is replaced by SPPFDino."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True, dino_mode="cls", dino_imgsz=224):
        super().__init__(cfg, ch=ch, nc=nc, verbose=False)
        self.fuse_idx = next(i for i, m in enumerate(self.model) if type(m) is SPPF)
        sppf = self.model[self.fuse_idx]
        self.model[self.fuse_idx] = SPPFDino(sppf, DINOv2Context(sppf.cv2.conv.out_channels, dino_mode, imgsz=dino_imgsz))
        self.dino_mode = dino_mode
        if verbose:
            self.info()

    def _predict_once(self, x, profile=False, embed=None):
        layer = self.model[self.fuse_idx] if hasattr(self, "fuse_idx") else None
        if layer is None:  # stride probe inside DetectionModel.__init__, before SPPF is swapped
            return super()._predict_once(x, profile, embed)
        layer.img = x
        try:
            return super()._predict_once(x, profile, embed)
        finally:
            layer.img = None  # never keep an image tensor on the module (it would be pickled into checkpoints)

    def _predict_augment(self, x):
        return self._predict_once(x)


class DinoDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that builds YOLODinoModel. Configure via class attributes before training."""

    dino_mode = "cls"
    dino_imgsz = 224

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLODinoModel(
            cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
            dino_mode=self.dino_mode,
            dino_imgsz=self.dino_imgsz,
        )
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model


def dino_freeze_spec(model: DetectionModel) -> list[str]:
    """Trainer `freeze` entries that freeze only the DINOv2 encoder (the projection stays trainable)."""
    idx = next(i for i, m in enumerate(model.model) if type(m) is SPPF)
    return [f"{idx}.dino.encoder"]
