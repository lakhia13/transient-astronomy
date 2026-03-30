"""
scripts/fix_csv_paths.py
------------------------
One-time migration: strip the absolute project-root prefix from path columns
in data/labels.csv and data/processed/split_labels.csv, leaving paths relative
to the project root (e.g. /data/raw/SN/ZTF18aayhvtx/521458333515015010_science.fits).
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREFIX = str(PROJECT_ROOT)

PATH_COLS_LABELS = ["science_path", "template_path", "difference_path"]
PATH_COLS_SPLIT  = PATH_COLS_LABELS + ["npy_path"]


def strip_prefix(val: str) -> str:
    if isinstance(val, str) and val.startswith(PREFIX):
        return val[len(PREFIX):]
    return val


def fix_csv(csv_path: Path, path_cols: list[str]) -> None:
    if not csv_path.exists():
        print(f"Skipping (not found): {csv_path}")
        return
    df = pd.read_csv(csv_path)
    for col in path_cols:
        if col in df.columns:
            df[col] = df[col].map(strip_prefix)
    df.to_csv(csv_path, index=False)
    print(f"Fixed {csv_path}  ({len(df)} rows)")


if __name__ == "__main__":
    fix_csv(PROJECT_ROOT / "data" / "labels.csv",            PATH_COLS_LABELS)
    fix_csv(PROJECT_ROOT / "data" / "processed" / "split_labels.csv", PATH_COLS_SPLIT)
