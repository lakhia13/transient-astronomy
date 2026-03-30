"""
src/train.py
------------
Training loop for the ZTF transient detection project (spec §8).

Public API
----------
train_epoch()    — one training epoch with mixed-precision AMP
evaluate_rb()    — validation pass returning AUC + AP (binary)
evaluate_mc()    — validation pass returning accuracy + per-class metrics (multiclass)
train()          — full training run with early stopping + checkpoint saving

Usage
-----
    from src.train import train

    history = train(
        model        = RealBogusResNet18(),
        train_loader = train_loader,
        val_loader   = val_loader,
        criterion    = FocalLoss(),
        task         = "real_bogus",
        epochs       = 50,
        checkpoint_dir = Path("checkpoints"),
    )
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One training epoch (spec §8)
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    task: str = "real_bogus",
    grad_clip: float = 1.0,
) -> dict:
    """
    Run one training epoch with FP16 mixed precision.

    Returns
    -------
    dict with keys: loss, n_batches, n_samples, elapsed_s
    """
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_samples = 0
    t0 = time.perf_counter()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(batch_x)
            if task == "real_bogus":
                loss = criterion(outputs, batch_y)
                preds = (torch.sigmoid(outputs) > 0.5).long()
            else:  # multiclass
                loss = criterion(outputs, batch_y)
                preds = outputs.argmax(dim=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_correct += (preds == batch_y).sum().item()
        n_samples += len(batch_y)

    elapsed = time.perf_counter() - t0
    return {
        "loss": total_loss / len(loader),
        "acc": n_correct / max(n_samples, 1),
        "n_batches": len(loader),
        "n_samples": n_samples,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Validation — binary real/bogus (spec §9)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rb(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    """
    Evaluate a binary (real/bogus) model.

    Returns
    -------
    dict: auc, ap, loss (if criterion given), n_samples
    """
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with autocast():
            logits = model(batch_x)
            if criterion is not None:
                total_loss += criterion(logits, batch_y).item()

        probs = torch.sigmoid(logits).cpu().float().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    result: dict = {
        "auc": float(roc_auc_score(all_labels_arr, all_probs_arr)),
        "ap": float(average_precision_score(all_labels_arr, all_probs_arr)),
        "n_samples": len(all_labels_arr),
        "probs": all_probs_arr,
        "labels": all_labels_arr,
    }
    if criterion is not None:
        result["loss"] = total_loss / max(len(loader), 1)
    return result


# ---------------------------------------------------------------------------
# Validation — multiclass transient type
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    num_classes: int = 5,
) -> dict:
    """
    Evaluate a multiclass model.

    Returns
    -------
    dict: accuracy, loss (if criterion), per-class probs and labels
    """
    model.eval()
    all_probs: list = []
    all_labels: list[int] = []
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with autocast():
            logits = model(batch_x)
            if criterion is not None:
                total_loss += criterion(logits, batch_y).item()

        probs = torch.softmax(logits, dim=1).cpu().float().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

    all_probs_arr = np.array(all_probs)   # (N, C)
    all_labels_arr = np.array(all_labels)  # (N,)
    preds = all_probs_arr.argmax(axis=1)

    result: dict = {
        "accuracy": float((preds == all_labels_arr).mean()),
        "n_samples": len(all_labels_arr),
        "probs": all_probs_arr,
        "labels": all_labels_arr,
        "preds": preds,
    }
    if criterion is not None:
        result["loss"] = total_loss / max(len(loader), 1)
    return result


# ---------------------------------------------------------------------------
# Full training run (spec §8)
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    task: str = "real_bogus",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    checkpoint_dir: Path | str = "checkpoints",
    run_name: str = "run",
    patience: int = 10,
    grad_clip: float = 1.0,
    scheduler_type: str = "cosine",
    wandb_run=None,
) -> dict:
    """
    Full training loop with:
    - AdamW optimiser
    - CosineAnnealingLR scheduler (spec §8)
    - Mixed precision (FP16)
    - Best checkpoint saving
    - Early stopping (patience)
    - Optional W&B logging

    Parameters
    ----------
    model         : nn.Module — model to train (will be moved to device)
    train_loader  : DataLoader for training split
    val_loader    : DataLoader for validation split
    criterion     : loss function
    task          : "real_bogus" or "multiclass"
    epochs        : maximum number of epochs
    lr            : initial learning rate (default 1e-3)
    weight_decay  : AdamW weight decay (default 1e-4)
    device        : torch.device (defaults to cuda if available)
    checkpoint_dir: directory where best model weights are saved
    run_name      : prefix for saved checkpoint filename
    patience      : early stopping patience (epochs without improvement)
    grad_clip     : gradient clipping max norm
    scheduler_type: "cosine" or "step" or "none"
    wandb_run     : optional W&B run object for metric logging

    Returns
    -------
    dict with keys:
        history    — list of per-epoch dicts (train_loss, val_metric, …)
        best_epoch — epoch index of the best checkpoint
        best_metric— best val AUC (real/bogus) or accuracy (multiclass)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    else:
        scheduler = None

    scaler = GradScaler()

    history: list[dict] = []
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    primary_key = "auc" if task == "real_bogus" else "accuracy"

    log.info(
        "Training %s | device=%s | epochs=%d | lr=%g | loss=%s",
        run_name, device, epochs, lr, criterion.__class__.__name__,
    )

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            task=task, grad_clip=grad_clip,
        )

        # ── Validate ───────────────────────────────────────────────────────
        if task == "real_bogus":
            val_stats = evaluate_rb(model, val_loader, device, criterion)
        else:
            val_stats = evaluate_mc(model, val_loader, device, criterion)

        val_metric = val_stats[primary_key]

        if scheduler is not None:
            scheduler.step()

        # ── Log ────────────────────────────────────────────────────────────
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            f"val_{primary_key}": val_metric,
            "val_loss": val_stats.get("loss", float("nan")),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if task == "real_bogus":
            epoch_log["val_ap"] = val_stats["ap"]

        history.append(epoch_log)

        log.info(
            "Epoch %3d | train_loss=%.4f | val_%s=%.4f | lr=%.2e",
            epoch, train_stats["loss"], primary_key, val_metric,
            optimizer.param_groups[0]["lr"],
        )

        if wandb_run is not None:
            wandb_run.log(epoch_log)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            no_improve = 0
            ckpt_path = checkpoint_dir / f"{run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    f"val_{primary_key}": val_metric,
                    "config": {"task": task, "lr": lr, "weight_decay": weight_decay},
                },
                ckpt_path,
            )
            log.info("  ✓ New best %.4f — saved to %s", best_metric, ckpt_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # Save training history as JSON
    history_path = checkpoint_dir / f"{run_name}_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint": str(checkpoint_dir / f"{run_name}_best.pt"),
    }


# ---------------------------------------------------------------------------
# Load checkpoint helper
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """
    Load weights from a checkpoint file into model in-place.

    Returns the full checkpoint dict (epoch, val metric, config).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", "?"))
    return ckpt
