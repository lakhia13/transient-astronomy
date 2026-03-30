"""
src/models.py
-------------
All model architectures for the ZTF transient detection project.

Spec §6 — four model families:
  1. RealBogusResNet18       — binary classifier (Task 1)
  2. TransientResNet18       — 5-class type classifier (Task 2)
  3. EfficientNetB0          — ablation alternative (via timm)
  4. ViTTiny                 — ablation alternative (via timm)

Each accepts (B, 3, 63, 63) float32 tensors.

Key adaptations for 63×63 input
---------------------------------
- conv1 changed from 7×7 stride-2 → 3×3 stride-1 to avoid excessive
  down-sampling of the small spatial dimensions.
- MaxPool after conv1 removed (replaced with nn.Identity) for the same reason.
- EfficientNet-B0 and ViT-Tiny are constructed via timm and work at 63×63
  without any structural changes (timm handles img_size).

Usage
-----
    from src.models import (
        RealBogusResNet18, TransientResNet18,
        EfficientNetB0, ViTTiny,
        build_model, count_parameters,
    )

    # Binary classifier
    model = build_model("resnet18_rb", task="real_bogus")
    # or
    model = RealBogusResNet18()
    logits = model(x)           # shape (B,)  — use torch.sigmoid to get P(real)

    # Multiclass classifier
    model = build_model("resnet18_mc", task="multiclass")
    logits = model(x)           # shape (B, 5) — use softmax for probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES: int = 5   # bogus, SN, AGN, VS, asteroid  (spec §6)
IMG_SIZE: int = 63     # native ZTF stamp resolution


# ---------------------------------------------------------------------------
# Shared helper — adapt ResNet-18 stem for small inputs
# ---------------------------------------------------------------------------

def _resnet18_small_input(num_output: int) -> nn.Module:
    """
    Build a ResNet-18 with a lightweight stem for 63×63 inputs.

    Changes from the default ImageNet ResNet-18:
    - conv1: 7×7 stride-2 → 3×3 stride-1 (keeps spatial resolution longer)
    - maxpool: removed (replaced with Identity)
    - fc: Dropout(0.4) + Linear(512 → num_output)
    """
    base = tvm.resnet18(weights=None)

    base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    base.maxpool = nn.Identity()  # type: ignore[assignment]

    base.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, num_output),
    )
    return base


# ---------------------------------------------------------------------------
# Task 1 — Real / Bogus Binary Classifier
# ---------------------------------------------------------------------------

class RealBogusResNet18(nn.Module):
    """
    Binary real/bogus classifier based on ResNet-18.

    Output: raw logit (scalar per sample, shape (B,)).
    Apply torch.sigmoid to obtain P(real ∈ [0, 1]).

    Loss: BCEWithLogitsLoss  or  FocalLoss (spec §7).
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = _resnet18_small_input(num_output=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)   # (B,)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method — return P(real) in [0, 1]."""
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Task 2 — Transient Type Multiclass Classifier
# ---------------------------------------------------------------------------

class TransientResNet18(nn.Module):
    """
    5-class transient type classifier based on ResNet-18.

    Classes (spec §3):  bogus=0, SN=1, AGN=2, VS=3, asteroid=4

    Output: raw logits of shape (B, NUM_CLASSES).
    Loss: CrossEntropyLoss (optionally with class weights, spec §7).

    In practice, this model is run only on detections that Task 1 classifies
    as 'real', but it is trained on all 5 classes including bogus so it
    can handle the full distribution at inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = _resnet18_small_input(num_output=NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, NUM_CLASSES)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax class probabilities, shape (B, NUM_CLASSES)."""
        return torch.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Alternative 1 — EfficientNet-B0 (ablation, spec §6 and §11 Ablation 2)
# ---------------------------------------------------------------------------

class EfficientNetB0(nn.Module):
    """
    EfficientNet-B0 wrapper via timm.

    Supports both binary and multiclass tasks via the `task` parameter.
    Input: (B, 3, 63, 63) float32.
    Binary  output: scalar logit (B,)   — use sigmoid.
    Multi   output: (B, NUM_CLASSES)    — use softmax.

    ~4.0M parameters — faster and lighter than ResNet-18 (11M).
    """

    def __init__(self, task: str = "real_bogus") -> None:
        super().__init__()
        assert task in ("real_bogus", "multiclass")
        self.task = task
        num_out = 1 if task == "real_bogus" else NUM_CLASSES
        self.net = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            in_chans=3,
            num_classes=num_out,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)          # (B, num_out)
        if self.task == "real_bogus":
            return out.squeeze(1)  # (B,)
        return out                 # (B, NUM_CLASSES)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == "real_bogus":
            return torch.sigmoid(self.forward(x))
        return torch.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Alternative 2 — ViT-Tiny (ablation, spec §6 and §11 Ablation 2)
# ---------------------------------------------------------------------------

class ViTTiny(nn.Module):
    """
    Vision Transformer Tiny (ViT-Tiny patch16) via timm.

    The model is configured with img_size=63 so the patch grid adapts to the
    native ZTF stamp resolution without resizing.  Patch size = 16 means
    each sequence has floor(63/16)^2 = 9 patches + 1 cls token = 10 tokens.

    Supports binary and multiclass tasks.
    ~5.5M parameters.
    """

    def __init__(self, task: str = "real_bogus") -> None:
        super().__init__()
        assert task in ("real_bogus", "multiclass")
        self.task = task
        num_out = 1 if task == "real_bogus" else NUM_CLASSES
        self.net = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            in_chans=3,
            num_classes=num_out,
            img_size=IMG_SIZE,      # native 63×63 — no resize needed
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)          # (B, num_out)
        if self.task == "real_bogus":
            return out.squeeze(1)  # (B,)
        return out                 # (B, NUM_CLASSES)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == "real_bogus":
            return torch.sigmoid(self.forward(x))
        return torch.softmax(self.forward(x), dim=1)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "resnet18_rb":       RealBogusResNet18,
    "resnet18_mc":       TransientResNet18,
    "efficientnet_b0":   EfficientNetB0,
    "vit_tiny":          ViTTiny,
}

_TASK_FOR_MODEL: dict[str, str] = {
    "resnet18_rb":       "real_bogus",
    "resnet18_mc":       "multiclass",
    "efficientnet_b0":   "real_bogus",   # default; overridable
    "vit_tiny":          "real_bogus",   # default; overridable
}


def build_model(
    name: str,
    task: str | None = None,
) -> nn.Module:
    """
    Instantiate a model by registry name.

    Parameters
    ----------
    name : str
        One of: "resnet18_rb", "resnet18_mc", "efficientnet_b0", "vit_tiny".
    task : str | None
        "real_bogus" or "multiclass".  For EfficientNet-B0 and ViT-Tiny this
        sets the output head.  Ignored for ResNet-18 variants (which have a
        fixed task baked in).  Defaults to the registry default for that name.

    Returns
    -------
    nn.Module
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.  Available: {list(_MODEL_REGISTRY)}"
        )
    cls = _MODEL_REGISTRY[name]
    effective_task = task or _TASK_FOR_MODEL[name]

    # ResNet-18 variants don't take a task arg (it's fixed by the class)
    if name in ("resnet18_rb", "resnet18_mc"):
        return cls()
    return cls(task=effective_task)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Return total and trainable parameter counts.

    Example
    -------
    >>> stats = count_parameters(model)
    >>> print(f"Total: {stats['total']/1e6:.2f}M  Trainable: {stats['trainable']/1e6:.2f}M")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary_table() -> list[dict]:
    """
    Build a summary of all models: params, input/output shape, VRAM estimate.
    Used in the architecture notebook (§6).
    """
    device = torch.device("cpu")
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    rows = []
    configs = [
        ("resnet18_rb",     None),
        ("resnet18_mc",     None),
        ("efficientnet_b0", "real_bogus"),
        ("efficientnet_b0", "multiclass"),
        ("vit_tiny",        "real_bogus"),
        ("vit_tiny",        "multiclass"),
    ]

    for name, task in configs:
        model = build_model(name, task).to(device).eval()
        with torch.no_grad():
            out = model(x)
        stats = count_parameters(model)
        rows.append({
            "name": name,
            "task": task or _TASK_FOR_MODEL[name],
            "params_M": round(stats["total"] / 1e6, 2),
            "input_shape": str(tuple(x.shape[1:])),
            "output_shape": str(tuple(out.shape[1:])),
            # Rough VRAM estimate: 4 bytes/param * 3 (params + grads + optim) + activations
            "vram_estimate_GB": round(stats["total"] * 4 * 3 / 1e9 + 0.3, 2),
        })

    return rows
