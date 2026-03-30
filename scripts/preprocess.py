#!/usr/bin/env python3
"""
scripts/preprocess.py
---------------------
Offline preprocessing pipeline — spec §5.

Reads data/labels.csv, applies the time-based 70/15/15 train/val/test
split, then for every triplet row:

  1. Loads the three FITS stamps (science, template, difference).
  2. Applies Z-score normalisation per channel (NaN/Inf → 0 first).
  3. Stacks into a (3, 63, 63) float32 numpy array.
  4. Saves to  data/processed/<split>/<candid>.npy

Also writes:
  data/processed/split_labels.csv  — the full labels CSV annotated with
                                      split, real_bogus_label, class_label
                                      and the .npy file path.
  data/processed/stats.json        — per-channel mean/std computed on the
                                      training set (used for optional global
                                      normalisation during training).

Usage
-----
    python scripts/preprocess.py [--limit N] [--workers W] [--force]

    --limit N      Process only the first N rows (for quick smoke-testing).
    --workers W    Parallel workers (default: min(8, cpu_count)).
    --force        Re-process even if the .npy already exists on disk.

Output
------
    data/processed/
    ├── train/          ← .npy files for training rows
    ├── val/            ← .npy files for validation rows
    ├── test/           ← .npy files for test rows
    ├── split_labels.csv
    └── stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Make src importable when the script is invoked from project root ─────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    CLASS_TO_IDX,
    REAL_BOGUS_MAP,
    load_triplet_arrays,
    prepare_triplet,
)
from src.dataset import time_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LABELS_CSV    = PROJECT_ROOT / "data" / "labels.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-row worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_row(args: tuple) -> dict:
    """
    Process a single labels-CSV row.

    Returns a status dict:
      { "candid": ..., "ok": bool, "error": str|None }
    """
    row_dict, out_dir, force = args
    candid    = str(row_dict["candid"])
    out_path  = Path(out_dir) / f"{candid}.npy"

    if not force and out_path.exists():
        return {"candid": candid, "ok": True, "error": None, "skipped": True}

    try:
        sci, tmpl, diff = load_triplet_arrays(
            PROJECT_ROOT / row_dict["science_path"].lstrip("/"),
            PROJECT_ROOT / row_dict["template_path"].lstrip("/"),
            PROJECT_ROOT / row_dict["difference_path"].lstrip("/"),
        )
        triplet = prepare_triplet(sci, tmpl, diff)   # (3, 63, 63) float32
        np.save(out_path, triplet)
        return {"candid": candid, "ok": True, "error": None, "skipped": False}
    except Exception as exc:
        return {"candid": candid, "ok": False, "error": str(exc), "skipped": False}


# ---------------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------------

def process_split(
    split_df: pd.DataFrame,
    split_name: str,
    workers: int,
    force: bool,
) -> tuple[int, int, int]:
    """
    Process all rows for one split.

    Returns (n_written, n_skipped, n_failed).
    """
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        (row.to_dict(), str(out_dir), force)
        for _, row in split_df.iterrows()
    ]

    n_written = n_skipped = n_failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_row, r): r[0]["candid"] for r in rows}

        with tqdm(
            total=len(futures),
            desc=f"  {split_name:5s}",
            unit="triplet",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result["ok"]:
                    if result.get("skipped"):
                        n_skipped += 1
                    else:
                        n_written += 1
                else:
                    n_failed += 1
                    log.debug("FAILED candid=%s  err=%s", result["candid"], result["error"])
                pbar.update(1)
                pbar.set_postfix(ok=n_written + n_skipped, fail=n_failed)

    return n_written, n_skipped, n_failed


# ---------------------------------------------------------------------------
# Per-channel stats (training set only)
# ---------------------------------------------------------------------------

def compute_channel_stats(
    train_df: pd.DataFrame,
    sample_n: int = 5000,
) -> dict:
    """
    Compute per-channel mean and std over a random sample of training triplets.

    The returned dict can be used to apply global normalisation on top of
    the per-image Z-score if desired.

    Channels: 0=science, 1=template, 2=difference.
    """
    sample = train_df.sample(min(sample_n, len(train_df)), random_state=42)
    npy_dir = PROCESSED_DIR / "train"

    all_pixels: list[list[float]] = [[], [], []]

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="  stats", unit="img"):
        npy_path = npy_dir / f"{row['candid']}.npy"
        if not npy_path.exists():
            continue
        arr = np.load(npy_path)          # (3, 63, 63)
        for ch in range(3):
            all_pixels[ch].extend(arr[ch].ravel().tolist())

    stats = {}
    for ch, name in enumerate(["science", "template", "difference"]):
        px = np.array(all_pixels[ch], dtype=np.float32)
        stats[name] = {
            "mean": float(px.mean()),
            "std":  float(px.std()),
        }
    return stats


# ---------------------------------------------------------------------------
# Split labels CSV
# ---------------------------------------------------------------------------

def write_split_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Write a single split_labels.csv with split + label columns added."""
    dfs = []
    for df, split in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        tmp = df.copy()
        tmp["split"]         = split
        tmp["real_bogus_label"] = tmp["class"].map(REAL_BOGUS_MAP)
        tmp["class_label"]      = tmp["class"].map(CLASS_TO_IDX)
        tmp["npy_path"]         = tmp["candid"].apply(
            lambda c: "/" + str((PROCESSED_DIR / split / f"{c}.npy").relative_to(PROJECT_ROOT))
        )
        dfs.append(tmp)

    out = pd.concat(dfs, ignore_index=True)
    out_path = PROCESSED_DIR / "split_labels.csv"
    out.to_csv(out_path, index=False)
    log.info("Wrote split_labels.csv → %s  (%d rows)", out_path, len(out))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess raw FITS stamps into normalised .npy triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (smoke-test mode).",
    )
    p.add_argument(
        "--workers", type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel worker processes.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-write .npy files even if they already exist.",
    )
    p.add_argument(
        "--train-frac", type=float, default=0.70,
        help="Fraction of data for training (chronological).",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Fraction of data for validation.",
    )
    p.add_argument(
        "--no-stats", action="store_true",
        help="Skip per-channel stats computation.",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Delete existing train/val/test split directories before processing. "
             "Implies --force. Use after adding new data to avoid stale .npy files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean:
        import shutil
        for split_name in ("train", "val", "test"):
            split_dir = PROCESSED_DIR / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
                log.info("Deleted %s", split_dir)
        args.force = True

    # ── Load labels ──────────────────────────────────────────────────────────
    if not LABELS_CSV.exists():
        log.error("labels.csv not found at %s. Run download_alerce.py first.", LABELS_CSV)
        sys.exit(1)

    log.info("Loading %s …", LABELS_CSV)
    df = pd.read_csv(LABELS_CSV)
    log.info("  %d rows, classes: %s", len(df), df["class"].value_counts().to_dict())

    if args.limit:
        df = df.head(args.limit)
        log.info("  (limited to first %d rows)", args.limit)

    # ── Time-based split ─────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = time_split(df, args.train_frac, args.val_frac)

    log.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    # ── Process each split ───────────────────────────────────────────────────
    total_written = total_skipped = total_failed = 0

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info("Processing '%s' split …", split_name)
        w, s, f = process_split(split_df, split_name, args.workers, args.force)
        total_written  += w
        total_skipped  += s
        total_failed   += f
        log.info(
            "  '%s' done — written: %d  skipped: %d  failed: %d",
            split_name, w, s, f,
        )

    # ── Per-channel stats ────────────────────────────────────────────────────
    if not args.no_stats:
        log.info("Computing per-channel statistics from training set …")
        stats = compute_channel_stats(train_df)
        stats_path = PROCESSED_DIR / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("  stats.json → %s", stats_path)
        for ch_name, vals in stats.items():
            log.info("  %-12s  mean=%+.4f  std=%.4f", ch_name, vals["mean"], vals["std"])

    # ── Split labels CSV ─────────────────────────────────────────────────────
    write_split_labels(train_df, val_df, test_df)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("Preprocessing complete")
    print("=" * 55)
    print(f"  Written  : {total_written:7,}")
    print(f"  Skipped  : {total_skipped:7,}  (already on disk)")
    print(f"  Failed   : {total_failed:7,}")
    print(f"  Output   : {PROCESSED_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
