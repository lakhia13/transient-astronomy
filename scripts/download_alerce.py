#!/usr/bin/env python3
"""
Download labeled training data from the ALeRCE ZTF broker.

Fetches object lists by class (stamp_classifier), then downloads
science/template/difference FITS stamps for each detection that has
a stamp available. Saves stamps to data/raw/<class>/<oid>/ and
writes a consolidated labels.csv to data/labels.csv.

Target volumes (from spec):
    bogus    : 15,000
    SN       :  3,000
    VS       :  3,000
    AGN      :  2,000
    asteroid :  1,000

Usage
-----
    python scripts/download_alerce.py [--dry-run] [--workers N]

Requirements
------------
    pip install requests tqdm astropy
"""

import argparse
import csv
import io
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from astropy.io import fits
except ImportError:
    fits = None  # stamps saved as raw bytes; FITS loading optional here

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api.alerce.online/ztf/v1"
STAMP_URL = "https://avro.alerce.online/get_stamp"

# Project root is two levels up from this script (scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_CSV = DATA_DIR / "labels.csv"

# Target number of objects to collect per class
CLASS_TARGETS: dict[str, int] = {
    "bogus": 15_000,
    "SN": 3_000,
    "VS": 3_000,
    "AGN": 2_000,
    "asteroid": 1_000,
}

STAMP_TYPES = ["science", "template", "difference"]
PAGE_SIZE = 1_000          # Max page_size accepted by ALeRCE API
REQUEST_TIMEOUT = 30       # seconds per HTTP request
RETRY_LIMIT = 3            # number of retries on transient errors
RETRY_BACKOFF = 2.0        # seconds between retries (doubles each attempt)
MIN_PROBABILITY = 0.5      # only keep objects above this classifier confidence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, stream: bool = False) -> requests.Response:
    """GET with retry logic and exponential back-off."""
    delay = RETRY_BACKOFF
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, stream=stream)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", delay))
                log.warning("Rate-limited. Sleeping %.1f s …", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == RETRY_LIMIT:
                raise
            log.debug("Request failed (%s), retry %d/%d in %.1f s", exc, attempt, RETRY_LIMIT, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# ALeRCE object listing
# ---------------------------------------------------------------------------

def fetch_objects_page(class_name: str, page: int) -> list[dict]:
    """Return one page of objects for *class_name* from the stamp classifier."""
    params = {
        "classifier": "stamp_classifier",
        "class": class_name,
        "probability": MIN_PROBABILITY,
        "page": page,
        "page_size": PAGE_SIZE,
        "order_by": "probability",
        "order_mode": "DESC",
    }
    data = _get(f"{BASE_URL}/objects/", params=params).json()
    return data.get("items", [])


def collect_objects(class_name: str, target: int) -> list[dict]:
    """
    Page through the ALeRCE objects endpoint until *target* objects are
    collected or the API is exhausted.  Returns a list of object dicts.
    """
    objects: list[dict] = []
    page = 1
    log.info("Collecting up to %d '%s' objects …", target, class_name)

    with tqdm(total=target, desc=f"  {class_name:8s} objects", unit="obj", leave=False) as pbar:
        while len(objects) < target:
            items = fetch_objects_page(class_name, page)
            if not items:
                log.info("  '%s': API exhausted at page %d (%d objects total)", class_name, page, len(objects))
                break
            before = len(objects)
            objects.extend(items[: target - len(objects)])
            pbar.update(len(objects) - before)
            if len(items) < PAGE_SIZE:
                # Last page
                break
            page += 1

    log.info("  '%s': collected %d objects", class_name, len(objects))
    return objects


# ---------------------------------------------------------------------------
# Detection listing
# ---------------------------------------------------------------------------

def fetch_detections(oid: str) -> list[dict]:
    """Return all detections for *oid* that carry stamps."""
    try:
        data = _get(f"{BASE_URL}/objects/{oid}/detections").json()
        return [d for d in data if d.get("has_stamp")]
    except Exception as exc:
        log.debug("fetch_detections(%s) failed: %s", oid, exc)
        return []


# ---------------------------------------------------------------------------
# Stamp downloading
# ---------------------------------------------------------------------------

def stamp_filepath(oid: str, candid: str, stamp_type: str, class_name: str) -> Path:
    """Return the destination path for a single stamp FITS file."""
    dest_dir = RAW_DIR / class_name / oid
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{candid}_{stamp_type}.fits"


def download_stamp(oid: str, candid: str, stamp_type: str, class_name: str) -> bool:
    """
    Download one stamp from avro.alerce.online and save it as a FITS file.
    Returns True on success, False on failure.
    """
    dest = stamp_filepath(oid, candid, stamp_type, class_name)
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already downloaded

    params = {
        "oid": oid,
        "candid": candid,
        "type": stamp_type,
        "format": "fits",
    }
    try:
        r = _get(STAMP_URL, params=params)
        dest.write_bytes(r.content)
        return True
    except Exception as exc:
        log.debug("stamp %s/%s/%s failed: %s", oid, candid, stamp_type, exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def download_triplet(oid: str, candid: str, class_name: str) -> tuple[bool, bool, bool]:
    """Download all three stamp types for one detection.  Returns (sci, tmpl, diff) booleans."""
    results = []
    for stamp_type in STAMP_TYPES:
        results.append(download_stamp(oid, candid, stamp_type, class_name))
    return tuple(results)


# ---------------------------------------------------------------------------
# Core download orchestration
# ---------------------------------------------------------------------------

def process_object(obj: dict, class_name: str, dry_run: bool) -> list[dict]:
    """
    For one ALeRCE object:
      1. Fetch its detections.
      2. Download the stamp triplet for each detection.
      3. Return a list of label rows (one per successful triplet).
    """
    oid = obj["oid"]
    rows: list[dict] = []

    detections = fetch_detections(oid)
    if not detections:
        return rows

    # One triplet per detection (multiple observations of same object)
    for det in detections:
        candid = str(det["candid"])
        mjd = det.get("mjd", "")

        if dry_run:
            rows.append({
                "oid": oid,
                "candid": candid,
                "class": class_name,
                "mjd": mjd,
                "science_path": "/" + str(stamp_filepath(oid, candid, "science", class_name).relative_to(PROJECT_ROOT)),
                "template_path": "/" + str(stamp_filepath(oid, candid, "template", class_name).relative_to(PROJECT_ROOT)),
                "difference_path": "/" + str(stamp_filepath(oid, candid, "difference", class_name).relative_to(PROJECT_ROOT)),
            })
            continue

        sci, tmpl, diff = download_triplet(oid, candid, class_name)
        if sci and tmpl and diff:
            rows.append({
                "oid": oid,
                "candid": candid,
                "class": class_name,
                "mjd": mjd,
                "science_path": "/" + str(stamp_filepath(oid, candid, "science", class_name).relative_to(PROJECT_ROOT)),
                "template_path": "/" + str(stamp_filepath(oid, candid, "template", class_name).relative_to(PROJECT_ROOT)),
                "difference_path": "/" + str(stamp_filepath(oid, candid, "difference", class_name).relative_to(PROJECT_ROOT)),
            })

    return rows


def download_class(class_name: str, target: int, workers: int, dry_run: bool) -> list[dict]:
    """Collect objects and download all their stamps for one class."""
    objects = collect_objects(class_name, target)
    all_rows: list[dict] = []

    log.info("Downloading stamps for %d '%s' objects (workers=%d) …", len(objects), class_name, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_object, obj, class_name, dry_run): obj["oid"]
            for obj in objects
        }
        with tqdm(total=len(futures), desc=f"  {class_name:8s} stamps", unit="obj") as pbar:
            for future in as_completed(futures):
                oid = futures[future]
                try:
                    rows = future.result()
                    all_rows.extend(rows)
                except Exception as exc:
                    log.warning("Object %s failed: %s", oid, exc)
                finally:
                    pbar.update(1)
                    pbar.set_postfix(triplets=len(all_rows))

    log.info("  '%s': %d triplets downloaded", class_name, len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Labels CSV
# ---------------------------------------------------------------------------

LABEL_COLUMNS = ["oid", "candid", "class", "mjd", "science_path", "template_path", "difference_path"]


def write_labels(rows: list[dict], path: Path) -> None:
    """Append rows to the labels CSV, writing header only if file is new."""
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download labeled ALeRCE/ZTF stamp triplets for ML training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(CLASS_TARGETS.keys()),
        choices=list(CLASS_TARGETS.keys()),
        help="Which classes to download.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=int,
        default=None,
        help="Override target counts (same order as --classes).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel download threads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without fetching stamps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-downloaded stamps (default: on).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build per-class target map from CLI overrides
    targets = dict(zip(args.classes, args.targets)) if args.targets else {c: CLASS_TARGETS[c] for c in args.classes}

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        log.info("DRY RUN — no files will be written")

    summary: dict[str, int] = {}

    for class_name in args.classes:
        target = targets[class_name]
        rows = download_class(class_name, target, args.workers, args.dry_run)
        if not args.dry_run:
            write_labels(rows, LABELS_CSV)
        summary[class_name] = len(rows)

    # Print summary
    print("\n" + "=" * 50)
    print("Download summary")
    print("=" * 50)
    total = 0
    for cls, count in summary.items():
        print(f"  {cls:10s}: {count:6d} triplets")
        total += count
    print(f"  {'TOTAL':10s}: {total:6d} triplets")
    if not args.dry_run:
        print(f"\n  Labels CSV : {LABELS_CSV}")
        print(f"  Stamp dir  : {RAW_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
