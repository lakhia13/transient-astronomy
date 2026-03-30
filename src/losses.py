"""
src/losses.py
-------------
Loss functions for the ZTF transient detection project (spec §7).

Three strategies for handling class imbalance:
  1. FocalLoss          — binary, down-weights easy examples
  2. WeightedBCELoss    — binary BCE with pos_weight scalar
  3. weighted_ce_loss() — helper to build weighted CrossEntropyLoss for multiclass

Usage
-----
    from src.losses import FocalLoss, WeightedBCELoss, make_weighted_ce

    # Binary real/bogus
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, targets)   # logits shape (B,), targets shape (B,) int

    # Multiclass with balanced class weights
    criterion = make_weighted_ce(train_labels, device)
    loss = criterion(logits, targets)   # logits (B, C), targets (B,) long
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Strategy 1 — Focal Loss (spec §7)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss (Focal Loss for Dense Object Detection, Lin et al. 2017).

    Focal loss = alpha * (1 - p_t)^gamma * BCE(logits, targets)

    Parameters
    ----------
    alpha : float
        Weighting factor for the rare (positive / real) class.
        Typical values: 0.25 – 0.75.  Lower alpha → less emphasis on positives.
    gamma : float
        Focusing parameter.  0 = standard BCE.  Higher gamma → more focus on
        hard misclassified examples.  Spec recommends 2.0.
    reduction : str
        "mean" (default), "sum", or "none".

    Input
    -----
    logits  : (B,)   — raw logits before sigmoid
    targets : (B,)   — integer labels {0, 1}
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        p_t = torch.exp(-bce)                               # probability of correct class
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction}"


# ---------------------------------------------------------------------------
# Strategy 1b — Focal Loss (multiclass variant)
# ---------------------------------------------------------------------------

class MulticlassFocalLoss(nn.Module):
    """
    Multiclass focal loss via cross-entropy.

    focal_weight = (1 - p_t)^gamma, applied element-wise on the CE loss.

    Parameters
    ----------
    gamma : float   — focusing parameter (0 = standard CE)
    weight : Tensor | None — optional per-class weights (shape (C,))
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)  # None-safe

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")  # (B,)

        # p_t = probability of the correct class
        with torch.no_grad():
            p_t = torch.exp(-ce)

        loss = (1.0 - p_t) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Strategy 2 — Weighted BCE (pos_weight scalar)
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with a scalar positive-class weight.

    pos_weight = n_neg / n_pos  (computed from training labels).
    Higher values push the model to recall more positives at the cost of precision.

    Parameters
    ----------
    pos_weight : float  — multiplier applied to loss for positive samples
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        w = torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", w)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight
        )

    @classmethod
    def from_labels(cls, labels: list[int]) -> "WeightedBCELoss":
        """Compute pos_weight = n_negative / n_positive from a label list."""
        labels_arr = np.asarray(labels)
        n_pos = int((labels_arr == 1).sum())
        n_neg = int((labels_arr == 0).sum())
        pos_weight = n_neg / max(n_pos, 1)
        return cls(pos_weight=pos_weight)


# ---------------------------------------------------------------------------
# Strategy 3 — Class-weighted CrossEntropyLoss (spec §7)
# ---------------------------------------------------------------------------

def make_weighted_ce(
    train_labels: list[int] | np.ndarray,
    device: torch.device,
    num_classes: int | None = None,
) -> nn.CrossEntropyLoss:
    """
    Build a CrossEntropyLoss with balanced class weights.

    Uses sklearn's 'balanced' strategy:
        weight_c = n_samples / (n_classes * count_c)

    Parameters
    ----------
    train_labels : array-like of int
        Integer labels from the training split.
    device : torch.device
        Device to place the weight tensor on.
    num_classes : int | None
        If given, ensures all classes 0…n-1 have a weight even if absent
        from train_labels.

    Returns
    -------
    nn.CrossEntropyLoss with pre-computed per-class weights.
    """
    labels_arr = np.asarray(train_labels)
    classes = np.arange(num_classes) if num_classes else np.unique(labels_arr)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr,
    )
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=w_tensor)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_loss(
    name: str,
    *,
    train_labels: list[int] | np.ndarray | None = None,
    device: torch.device = torch.device("cpu"),
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    num_classes: int | None = None,
) -> nn.Module:
    """
    Instantiate a loss by name.

    Names
    -----
    "focal"           — FocalLoss (binary)
    "bce"             — plain BCEWithLogitsLoss
    "weighted_bce"    — WeightedBCELoss (requires train_labels)
    "ce"              — plain CrossEntropyLoss (multiclass)
    "weighted_ce"     — class-weighted CE (requires train_labels)
    "focal_mc"        — MulticlassFocalLoss
    "weighted_focal_mc" — MulticlassFocalLoss + balanced class weights
    """
    name = name.lower()

    if name == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    if name == "bce":
        return nn.BCEWithLogitsLoss()

    if name == "weighted_bce":
        assert train_labels is not None, "train_labels required for weighted_bce"
        return WeightedBCELoss.from_labels(train_labels)

    if name == "ce":
        return nn.CrossEntropyLoss()

    if name == "weighted_ce":
        assert train_labels is not None, "train_labels required for weighted_ce"
        return make_weighted_ce(train_labels, device, num_classes)

    if name == "focal_mc":
        return MulticlassFocalLoss(gamma=focal_gamma)

    if name == "weighted_focal_mc":
        assert train_labels is not None
        labels_arr = np.asarray(train_labels)
        classes = np.arange(num_classes) if num_classes else np.unique(labels_arr)
        weights = compute_class_weight("balanced", classes=classes, y=labels_arr)
        w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        return MulticlassFocalLoss(gamma=focal_gamma, weight=w_tensor)

    raise ValueError(
        f"Unknown loss '{name}'.  Available: focal, bce, weighted_bce, "
        "ce, weighted_ce, focal_mc, weighted_focal_mc"
    )
