#!/usr/bin/env python3
"""
Upload the ZTF transient stamp dataset to the Hugging Face Hub.

What is uploaded
----------------
The entire data/ directory is uploaded as-is:

    data/
    ├── labels.csv
    ├── raw/
    │   └── <class>/<oid>/<candid>_science|template|difference.fits
    └── processed/
        ├── split_labels.csv
        ├── stats.json
        └── train|val|test/<candid>.npy

Usage
-----
    # Public repo
    python scripts/upload_huggingface.py --repo YOUR_HF_USERNAME/ztf-transient-stamps

    # Private repo
    python scripts/upload_huggingface.py --repo YOUR_HF_USERNAME/ztf-transient-stamps --private

Authentication
--------------
    Log in once before running:
        huggingface-cli login
    or set the HF_TOKEN environment variable.

Requirements
------------
    pip install huggingface_hub
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_prerequisites() -> None:
    """Fail fast with a clear message if required files are missing."""
    missing = []
    if not DATA_DIR.exists():
        missing.append(str(DATA_DIR))
    if not (DATA_DIR / "labels.csv").exists():
        missing.append(str(DATA_DIR / "labels.csv"))
    if not (DATA_DIR / "processed" / "split_labels.csv").exists():
        missing.append(str(DATA_DIR / "processed" / "split_labels.csv"))
    if missing:
        log.error(
            "The following required paths are missing:\n  %s\n"
            "Run scripts/download_alerce.py and scripts/preprocess.py first.",
            "\n  ".join(missing),
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload the full ZTF data/ folder to Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Hub repo ID in the form USERNAME/DATASET-NAME (e.g. jdoe/ztf-transient-stamps).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token. Defaults to HF_TOKEN env var or cached login.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.error("'huggingface_hub' package not found.  Run: pip install huggingface_hub")
        sys.exit(1)

    _check_prerequisites()

    token = args.token or os.environ.get("HF_TOKEN") or None
    api   = HfApi(token=token)

    # ── Create repo if it doesn't exist ────────────────────────────────────
    log.info("Creating/verifying repository: %s (private=%s) …", args.repo, args.private)
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # ── Upload data/ folder ─────────────────────────────────────────────────
    log.info("Uploading %s (%.1f GB) — this may take a while …",
             DATA_DIR, sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) / 1e9)

    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(DATA_DIR),
    )

    log.info("Upload complete.")
    log.info("Dataset URL: https://huggingface.co/datasets/%s", args.repo)


if __name__ == "__main__":
    main()
