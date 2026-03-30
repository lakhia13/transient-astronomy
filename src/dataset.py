"""
src/dataset.py
--------------
PyTorch Dataset class and dataset-building utilities for the transient
detection project.

Key public API
--------------
TransientDataset       – torch Dataset that reads pre-processed .npy triplets
                         or loads raw FITS on-the-fly.
build_datasets()       – Given a labels DataFrame, apply the time-based
                         70/15/15 split and return (train, val, test) datasets.
make_weighted_sampler()– WeightedRandomSampler for handling class imbalance.
get_augmentation()     – albumentations pipeline for training-time augmentation.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import albumentations as A

from src.utils import (
    CLASS_TO_IDX,
    REAL_BOGUS_MAP,
    load_triplet_arrays,
    prepare_triplet,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str) -> Path:
    """Resolve a project-root-relative path (e.g. /data/raw/...) to an absolute Path."""
    return PROJECT_ROOT / path.lstrip("/")

# ---------------------------------------------------------------------------
# Augmentation (spec §5 — training only, no aug on val / test)
# ---------------------------------------------------------------------------

def get_augmentation(split: str = "train") -> Optional[A.Compose]:
    """
    Return an albumentations Compose pipeline.

    Training:  horizontal flip + vertical flip + rotation + gaussian noise.
    Val / Test: None.

    The input to albumentations is (H, W, C) uint8 or float32.
    We feed (63, 63, 3) float32, so we wrap with a channel transpose.
    """
    if split != "train":
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.5, border_mode=0),   # no preferred orientation in astronomy
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransientDataset(Dataset):
    """
    Triplet dataset for ZTF transient detection.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of labels.csv for this split.  Must contain columns:
        science_path, template_path, difference_path, class, mjd.
    task : {"real_bogus", "multiclass"}
        "real_bogus"  → binary label  0=bogus / 1=real
        "multiclass"  → integer label 0…4  (CLASS_TO_IDX order)
    processed_dir : Path | None
        If provided, looks for pre-cached .npy triplets in
        processed_dir/<split>/<candid>.npy  before falling back to raw FITS.
        Set to None to always load from raw FITS.
    transform : albumentations.Compose | None
        Image augmentation pipeline.  If None, no augmentation is applied.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str = "real_bogus",
        processed_dir: Optional[Path] = None,
        transform: Optional[A.Compose] = None,
    ):
        assert task in ("real_bogus", "multiclass"), f"Unknown task: {task}"
        self.df = df.reset_index(drop=True)
        self.task = task
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.transform = transform

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 1. Load triplet ------------------------------------------------
        triplet = self._load_triplet(row)  # (3, 63, 63) float32 numpy

        # 2. Augmentation ------------------------------------------------
        if self.transform is not None:
            # albumentations expects (H, W, C) — transpose, augment, transpose back
            hwc = triplet.transpose(1, 2, 0)       # (63, 63, 3)
            hwc = self.transform(image=hwc)["image"]
            triplet = hwc.transpose(2, 0, 1)       # (3, 63, 63)

        # 3. Label -------------------------------------------------------
        label = self._get_label(row)

        x = torch.tensor(triplet, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    # ------------------------------------------------------------------
    def _load_triplet(self, row) -> np.ndarray:
        """Try cached .npy first; fall back to raw FITS."""
        if self.processed_dir is not None:
            npy_path = self.processed_dir / f"{row['candid']}.npy"
            if npy_path.exists():
                return np.load(npy_path)

        sci, tmpl, diff = load_triplet_arrays(
            _resolve(row["science_path"]),
            _resolve(row["template_path"]),
            _resolve(row["difference_path"]),
        )
        return prepare_triplet(sci, tmpl, diff)

    # ------------------------------------------------------------------
    def _get_label(self, row) -> int:
        if self.task == "real_bogus":
            return REAL_BOGUS_MAP[row["class"]]
        return CLASS_TO_IDX[row["class"]]

    # ------------------------------------------------------------------
    @property
    def labels(self) -> list[int]:
        """All integer labels in dataset order.  Used by WeightedRandomSampler."""
        return [self._get_label(self.df.iloc[i]) for i in range(len(self))]


# ---------------------------------------------------------------------------
# Time-based split (spec §5 — DO NOT use random split)
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by MJD and split chronologically.

    train : oldest  train_frac  of rows
    val   : next    val_frac    of rows
    test  : remaining            rows  (= 1 - train_frac - val_frac)

    This ensures the model is always evaluated on *future* data and
    prevents temporal data leakage (spec §5).
    """
    df_sorted = df.sort_values("mjd").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_df = df_sorted.iloc[:n_train]
    val_df   = df_sorted.iloc[n_train : n_train + n_val]
    test_df  = df_sorted.iloc[n_train + n_val :]

    log.info(
        "Time split: train=%d  val=%d  test=%d  "
        "(MJD %.1f – %.1f / %.1f – %.1f / %.1f – %.1f)",
        len(train_df), len(val_df), len(test_df),
        train_df["mjd"].min(), train_df["mjd"].max(),
        val_df["mjd"].min(),   val_df["mjd"].max(),
        test_df["mjd"].min(),  test_df["mjd"].max(),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Weighted random sampler (spec §7 Strategy 2)
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each batch sees a balanced class view.

    Weight of each sample = 1 / count(class).
    """
    counts = Counter(labels)
    weights = [1.0 / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_datasets(
    df: pd.DataFrame,
    task: str = "real_bogus",
    processed_dir: Optional[Path] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[TransientDataset, TransientDataset, TransientDataset]:
    """
    Apply the time-based split and wrap each partition in a TransientDataset.

    Returns (train_ds, val_ds, test_ds).
    Training dataset includes albumentations augmentation; val/test do not.
    """
    train_df, val_df, test_df = time_split(df, train_frac, val_frac)

    train_ds = TransientDataset(
        train_df,
        task=task,
        processed_dir=processed_dir,
        transform=get_augmentation("train"),
    )
    val_ds = TransientDataset(
        val_df,
        task=task,
        processed_dir=processed_dir,
        transform=None,
    )
    test_ds = TransientDataset(
        test_df,
        task=task,
        processed_dir=processed_dir,
        transform=None,
    )

    log.info(
        "Datasets built — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
