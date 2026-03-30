"""
src/evaluate.py
---------------
Evaluation utilities for the ZTF transient detection project (spec §9).

Functions
---------
full_eval_rb()          — compute all binary metrics (AUC, AP, F1, confusion matrix)
full_eval_mc()          — compute all multiclass metrics (accuracy, per-class P/R/F1)
threshold_analysis()    — sweep thresholds and find the recall≥0.95 operating point
plot_roc()              — ROC curve (supports multiple models on same axes)
plot_pr_curve()         — Precision-Recall curve
plot_confusion_matrix() — annotated heat-map confusion matrix
plot_threshold_curve()  — precision, recall, F1 vs threshold

Usage
-----
    from src.evaluate import full_eval_rb, plot_roc, threshold_analysis

    metrics = full_eval_rb(all_probs, all_labels)
    print(metrics['auc'], metrics['ap'])
    plot_roc({'ResNet-18': (all_probs, all_labels)})
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Binary evaluation (spec §9)
# ---------------------------------------------------------------------------

def full_eval_rb(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Full binary evaluation suite.

    Parameters
    ----------
    probs   : (N,) float — predicted P(real) in [0, 1]
    labels  : (N,) int  — ground-truth {0=bogus, 1=real}
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict with: auc, ap, f1, precision, recall, accuracy, confusion_matrix,
               classification_report, threshold
    """
    preds = (probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(labels, probs))
    ap = float(average_precision_score(labels, probs))
    f1 = float(f1_score(labels, preds, zero_division=0))

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(len(labels), 1)

    return {
        "auc": roc_auc,
        "ap": ap,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": confusion_matrix(labels, preds),
        "classification_report": classification_report(
            labels, preds, target_names=["bogus", "real"], zero_division=0
        ),
        "threshold": threshold,
        "n_samples": len(labels),
    }


# ---------------------------------------------------------------------------
# Multiclass evaluation
# ---------------------------------------------------------------------------

def full_eval_mc(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Full multiclass evaluation suite.

    Parameters
    ----------
    probs       : (N, C) float — softmax probabilities
    labels      : (N,) int    — ground-truth class indices
    class_names : list of C strings

    Returns
    -------
    dict with: accuracy, per-class precision/recall/f1, confusion_matrix,
               macro_f1, weighted_f1, classification_report
    """
    preds = probs.argmax(axis=1)
    accuracy = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(labels),
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Threshold analysis (spec §9)
# ---------------------------------------------------------------------------

def threshold_analysis(
    probs: np.ndarray,
    labels: np.ndarray,
    target_recall: float = 0.95,
) -> dict:
    """
    Sweep decision thresholds and find the operating point that achieves
    recall >= target_recall with highest precision.

    Parameters
    ----------
    probs          : (N,) float — predicted P(real)
    labels         : (N,) int  — ground-truth {0, 1}
    target_recall  : minimum recall required (astronomers prefer high recall)

    Returns
    -------
    dict with: precision_curve, recall_curve, thresholds, f1_curve,
               operating_threshold, operating_precision, operating_recall, operating_f1
    """
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probs)

    # F1 at each threshold (length = len(thresholds), not len(precision_curve))
    f1_curve = np.where(
        (precision_curve[:-1] + recall_curve[:-1]) > 0,
        2 * precision_curve[:-1] * recall_curve[:-1]
        / (precision_curve[:-1] + recall_curve[:-1]),
        0.0,
    )

    # Find threshold that achieves target recall with highest precision
    mask = recall_curve[:-1] >= target_recall
    if mask.any():
        best_idx = int(np.argmax(precision_curve[:-1][mask]))
        indices = np.where(mask)[0]
        operating_idx = indices[best_idx]
        op_threshold = float(thresholds[operating_idx])
        op_precision = float(precision_curve[operating_idx])
        op_recall = float(recall_curve[operating_idx])
        op_f1 = float(f1_curve[operating_idx])
    else:
        # Fallback: threshold with highest F1
        operating_idx = int(np.argmax(f1_curve))
        op_threshold = float(thresholds[operating_idx])
        op_precision = float(precision_curve[operating_idx])
        op_recall = float(recall_curve[operating_idx])
        op_f1 = float(f1_curve[operating_idx])

    return {
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "thresholds": thresholds,
        "f1_curve": f1_curve,
        "operating_threshold": op_threshold,
        "operating_precision": op_precision,
        "operating_recall": op_recall,
        "operating_f1": op_f1,
        "target_recall": target_recall,
    }


# ---------------------------------------------------------------------------
# Plotting helpers (spec §15 visualisations)
# ---------------------------------------------------------------------------

def plot_roc(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: plt.Axes | None = None,
    title: str = "ROC Curves",
    save_path: str | None = None,
) -> plt.Axes:
    """
    Plot ROC curves for one or more models on the same axes.

    Parameters
    ----------
    models : dict mapping model_name → (probs, labels)
    ax     : existing Axes to draw on (creates new figure if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (AUC=0.50)")

    for (name, (probs, labels)), color in zip(models.items(), colors):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{name}  (AUC={roc_auc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    return ax


def plot_pr_curve(
    models: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ax: plt.Axes | None = None,
    title: str = "Precision-Recall Curves",
    save_path: str | None = None,
) -> plt.Axes:
    """Plot Precision-Recall curves for one or more models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for (name, (probs, labels)), color in zip(models.items(), colors):
        p, r, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(r, p, linewidth=2, color=color,
                label=f"{name}  (AP={ap:.4f})")

    # Baseline (random classifier)
    pos_rate = float(np.mean(list(models.values())[0][1]))
    ax.axhline(pos_rate, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"Random (AP={pos_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    return ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    ax: plt.Axes | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    save_path: str | None = None,
) -> plt.Axes:
    """
    Plot an annotated confusion matrix heat-map.

    Parameters
    ----------
    cm          : raw (unnormalized) confusion matrix from sklearn
    class_names : list of class name strings
    normalize   : if True, show row-normalized percentages
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(class_names)), max(4, len(class_names))))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = cm.astype(float) / np.maximum(row_sums, 1)
        fmt = ".2f"
    else:
        cm_plot = cm.astype(float)
        fmt = ".0f"

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=1 if normalize else cm_plot.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm_plot.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            raw = cm[i, j]
            val = cm_plot[i, j]
            text = f"{val:{fmt}}\n({raw})" if normalize else f"{raw:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    return ax


def plot_threshold_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes | None = None,
    target_recall: float = 0.95,
    title: str = "Precision / Recall / F1 vs Threshold",
    save_path: str | None = None,
) -> plt.Axes:
    """
    Plot precision, recall, and F1 as a function of the decision threshold.
    Mark the operating point that achieves target_recall.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    result = threshold_analysis(probs, labels, target_recall=target_recall)
    thresholds = result["thresholds"]
    p_curve = result["precision_curve"][:-1]
    r_curve = result["recall_curve"][:-1]
    f1_c = result["f1_curve"]

    ax.plot(thresholds, p_curve, color="#2196F3", linewidth=2, label="Precision")
    ax.plot(thresholds, r_curve, color="#4CAF50", linewidth=2, label="Recall")
    ax.plot(thresholds, f1_c, color="#FF9800", linewidth=2, label="F1")

    op = result["operating_threshold"]
    ax.axvline(op, color="#E91E63", linestyle="--", linewidth=1.5,
               label=f"Op. point (t={op:.3f})\nP={result['operating_precision']:.3f} "
                     f"R={result['operating_recall']:.3f}")
    ax.axhline(target_recall, color="gray", linestyle=":", linewidth=1,
               alpha=0.7, label=f"Target recall={target_recall}")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    return ax


def plot_training_curves(
    history: list[dict],
    task: str = "real_bogus",
    ax: tuple | None = None,
    save_path: str | None = None,
) -> tuple:
    """
    Plot training loss and validation AUC/accuracy per epoch.

    Parameters
    ----------
    history : list of epoch dicts from train()
    task    : "real_bogus" or "multiclass"
    ax      : (ax_loss, ax_metric) tuple or None
    """
    if ax is None:
        fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))
    else:
        ax_loss, ax_metric = ax

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h.get("val_loss", float("nan")) for h in history]
    primary_key = "auc" if task == "real_bogus" else "accuracy"
    val_metric = [h.get(f"val_{primary_key}", float("nan")) for h in history]

    ax_loss.plot(epochs, train_loss, "o-", color="#2196F3", linewidth=2,
                 markersize=4, label="Train loss")
    ax_loss.plot(epochs, val_loss, "s--", color="#FF9800", linewidth=2,
                 markersize=4, label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend(fontsize=8)

    label = "Val AUC-ROC" if task == "real_bogus" else "Val Accuracy"
    ax_metric.plot(epochs, val_metric, "o-", color="#4CAF50", linewidth=2,
                   markersize=4)
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_ylabel(label)
    ax_metric.set_title(label + " per Epoch")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    return ax_loss, ax_metric
