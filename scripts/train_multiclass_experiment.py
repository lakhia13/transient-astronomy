"""
Starter script for isolated multiclass experiments.

This script intentionally writes outputs under:
  - checkpoints/experiments/<run_name>/
  - results/experiments/<run_name>/
so existing project checkpoints/results remain untouched.

After training, it runs a full evaluation on the test set and saves:
  - summary.json            Top-line metrics + config
  - full_eval.json          Per-class precision/recall/F1, confusion matrix
  - confusion_matrix.png    Row-normalised confusion matrix heatmap
  - confusion_matrix_raw.png Raw-count confusion matrix
  - training_curves.png     Train/val loss and val accuracy per epoch
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so PNGs save cleanly on headless GPUs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import build_datasets, make_weighted_sampler
from src.evaluate import (
    full_eval_mc,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.losses import build_loss
from src.models import build_model
from src.train import evaluate_mc, load_checkpoint, train
from src.utils import IDX_TO_CLASS


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_from_project(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides onto the loaded YAML config (in-place copy).

    Supported overrides:
      --epochs N         → training.epochs
      --run-name NAME    → experiment.name
      --smoke-test       → training.epochs=1, training.patience=1, training.num_workers=2
                           and appends '_smoke' to the run name so it doesn't clobber
                           a real run.
    """
    training = cfg.setdefault("training", {})
    experiment = cfg.setdefault("experiment", {})

    if args.smoke_test:
        training["epochs"] = 1
        training["patience"] = 1
        training["num_workers"] = min(int(training.get("num_workers", 4)), 2)
        base_name = experiment.get("name", "run")
        if not base_name.endswith("_smoke"):
            experiment["name"] = f"{base_name}_smoke"

    if args.epochs is not None:
        training["epochs"] = int(args.epochs)

    if args.run_name is not None:
        experiment["name"] = args.run_name

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated multiclass experiment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multiclass_exp_v1.yaml"),
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs from the config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override experiment.name from the config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick 1-epoch sanity run (writes to <name>_smoke). "
             "Use this first in Colab to verify the pipeline works.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _apply_overrides(_load_config(args.config), args)

    run_name = cfg["experiment"]["name"]
    seed = int(cfg["experiment"].get("seed", 42))
    _set_seed(seed)

    labels_csv = _resolve_from_project(cfg["paths"]["labels_csv"])
    processed_dir = _resolve_from_project(cfg["paths"]["processed_dir"])
    ckpt_root = _resolve_from_project(cfg["paths"]["checkpoint_root"])
    results_root = _resolve_from_project(cfg["paths"]["results_root"])

    if not labels_csv.exists():
        raise FileNotFoundError(
            f"labels_csv not found: {labels_csv}\n"
            "Generate data first (download + preprocess) or point config paths to "
            "an existing dataset location."
        )
    if not processed_dir.exists():
        raise FileNotFoundError(
            f"processed_dir not found: {processed_dir}\n"
            "Expected preprocessed .npy data at data/processed. Run preprocessing or "
            "update the config path."
        )

    run_ckpt_dir = ckpt_root / run_name
    run_results_dir = results_root / run_name

    # Allow overwrite when it's a smoke-test run, since those are disposable.
    allow_overwrite = args.overwrite or args.smoke_test
    if (run_ckpt_dir.exists() or run_results_dir.exists()) and not allow_overwrite:
        raise FileExistsError(
            f"Run directories already exist for '{run_name}'. "
            "Use a new run name or pass --overwrite."
        )

    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] run_name={run_name} | device={device} | seed={seed}")

    df = pd.read_csv(labels_csv)
    train_ds, val_ds, test_ds = build_datasets(
        df=df,
        task="multiclass",
        processed_dir=processed_dir,
    )
    print(
        f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    training_cfg = cfg["training"]
    batch_size = int(training_cfg.get("batch_size", 64))
    num_workers = int(training_cfg.get("num_workers", 4))
    use_weighted_sampler = bool(training_cfg.get("use_weighted_sampler", True))

    if use_weighted_sampler:
        sampler = make_weighted_sampler(train_ds.labels)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    model = build_model(training_cfg.get("model", "resnet18_mc"), task="multiclass")
    criterion = build_loss(
        name=training_cfg.get("loss", "weighted_ce"),
        train_labels=train_ds.labels,
        device=device,
        num_classes=len(IDX_TO_CLASS),
    )

    train_out = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        task="multiclass",
        epochs=int(training_cfg.get("epochs", 30)),
        lr=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        device=device,
        checkpoint_dir=run_ckpt_dir,
        run_name=run_name,
        patience=int(training_cfg.get("patience", 8)),
        grad_clip=float(training_cfg.get("grad_clip", 1.0)),
        scheduler_type=training_cfg.get("scheduler", "cosine"),
    )

    # ── Final evaluation on the test set ──────────────────────────────────
    best_ckpt = run_ckpt_dir / f"{run_name}_best.pt"
    load_checkpoint(model, best_ckpt)

    # evaluate_mc returns probs/labels; full_eval_mc crunches them into
    # per-class metrics, macro-F1, confusion matrix, etc.
    raw = evaluate_mc(model=model, loader=test_loader, device=device, criterion=None)
    class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]
    eval_out = full_eval_mc(
        probs=raw["probs"],
        labels=raw["labels"],
        class_names=class_names,
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    cm = eval_out["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        ax=ax,
        title=f"{run_name} — Confusion Matrix (row-normalised)",
        normalize=True,
    )
    fig.tight_layout()
    fig.savefig(run_results_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        ax=ax,
        title=f"{run_name} — Confusion Matrix (raw counts)",
        normalize=False,
    )
    fig.tight_layout()
    fig.savefig(run_results_dir / "confusion_matrix_raw.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    history = train_out.get("history", [])
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plot_training_curves(history, task="multiclass", ax=tuple(axes))
        fig.suptitle(f"{run_name} — Training Curves")
        fig.tight_layout()
        fig.savefig(run_results_dir / "training_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ── JSON outputs ──────────────────────────────────────────────────────
    summary = {
        "run_name": run_name,
        "seed": seed,
        "device": str(device),
        "data": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "train": {
            "best_epoch": train_out["best_epoch"],
            "best_val_metric": train_out["best_metric"],
            "checkpoint": train_out["checkpoint"],
        },
        "test": {
            "accuracy": eval_out["accuracy"],
            "macro_f1": eval_out["macro_f1"],
            "weighted_f1": eval_out["weighted_f1"],
            "per_class_metrics": eval_out["per_class_metrics"],
        },
        "config": cfg,
        "class_names": class_names,
    }

    (run_results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # full_eval_mc returns a numpy confusion matrix — convert for JSON.
    full_eval_serializable = {
        "accuracy": eval_out["accuracy"],
        "macro_f1": eval_out["macro_f1"],
        "weighted_f1": eval_out["weighted_f1"],
        "n_samples": eval_out["n_samples"],
        "class_names": eval_out["class_names"],
        "per_class_metrics": eval_out["per_class_metrics"],
        "confusion_matrix": eval_out["confusion_matrix"].tolist(),
        "classification_report": eval_out["classification_report"],
    }
    (run_results_dir / "full_eval.json").write_text(
        json.dumps(full_eval_serializable, indent=2), encoding="utf-8"
    )

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n[done] Run: {run_name}")
    print(f"[done] Checkpoints: {run_ckpt_dir}")
    print(f"[done] Results:     {run_results_dir}")
    print(f"[done] Test accuracy:   {eval_out['accuracy']:.4f}")
    print(f"[done] Test macro-F1:   {eval_out['macro_f1']:.4f}")
    print(f"[done] Test weighted-F1: {eval_out['weighted_f1']:.4f}")
    print("\n[done] Classification report:")
    print(eval_out["classification_report"])


if __name__ == "__main__":
    main()
