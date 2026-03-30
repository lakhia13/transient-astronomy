"""
src/utils.py
------------
Low-level helpers used across the preprocessing pipeline and training code.

Responsibilities
----------------
- Load a single FITS stamp from disk → float32 numpy array
- Z-score normalise one image (with NaN/Inf protection)
- Stack science + template + difference → (3, 63, 63) numpy array
- Convert that stack to a float32 torch tensor
- Class-label encode/decode utilities
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Torch is imported lazily so this module can be used without GPU support
# (e.g. in the offline preprocessing script that runs on CPU).
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = ["bogus", "SN", "AGN", "VS", "asteroid"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# Binary real/bogus: 0 = bogus, 1 = real
REAL_BOGUS_MAP: dict[str, int] = {c: (0 if c == "bogus" else 1) for c in CLASS_NAMES}


def class_to_idx(class_name: str) -> int:
    """Map a class name string to its integer index."""
    if class_name not in CLASS_TO_IDX:
        raise ValueError(f"Unknown class '{class_name}'. Valid: {CLASS_NAMES}")
    return CLASS_TO_IDX[class_name]


def idx_to_class(idx: int) -> str:
    return IDX_TO_CLASS[idx]


# ---------------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------------

def load_fits(path: str | Path) -> np.ndarray:
    """
    Load one FITS stamp and return a float32 (H, W) numpy array.

    Handles:
    - Big-endian dtype ('>f4') from FITS — converted to native float32.
    - Missing file — returns a zero array of shape (63, 63).
    """
    from astropy.io import fits  # deferred import; heavy at module level

    path = Path(path)
    if not path.exists():
        return np.zeros((63, 63), dtype=np.float32)
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
    if data is None:
        return np.zeros((63, 63), dtype=np.float32)
    arr = data.astype(np.float32)
    # Ensure exactly (63, 63) — some stamps are non-standard sizes
    if arr.shape != (63, 63):
        arr = _resize_to_63(arr)
    return arr


def _resize_to_63(arr: np.ndarray, target: int = 63) -> np.ndarray:
    """Centre-crop or zero-pad a 2D array to (target, target)."""
    h, w = arr.shape
    out = np.zeros((target, target), dtype=np.float32)
    # Determine crop/pad offsets
    # Row
    if h >= target:
        r0 = (h - target) // 2
        src_r = slice(r0, r0 + target)
        dst_r = slice(0, target)
    else:
        src_r = slice(0, h)
        pad = (target - h) // 2
        dst_r = slice(pad, pad + h)
    # Col
    if w >= target:
        c0 = (w - target) // 2
        src_c = slice(c0, c0 + target)
        dst_c = slice(0, target)
    else:
        src_c = slice(0, w)
        pad = (target - w) // 2
        dst_c = slice(pad, pad + w)
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


# ---------------------------------------------------------------------------
# Image normalisation (spec §5)
# ---------------------------------------------------------------------------

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Z-score normalisation with NaN / Inf protection.

    Steps
    -----
    1. Replace NaN, +Inf, -Inf with 0.
    2. Cast to float64 for numerically stable mean/std computation
       (raw FITS ADU values can overflow float32 during summation).
    3. Subtract mean, divide by std.
    4. Return float32.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img64 = img.astype(np.float64)
    mean = img64.mean()
    std  = img64.std()
    if std == 0.0:
        return (img64 - mean).astype(np.float32)
    return ((img64 - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Triplet helpers
# ---------------------------------------------------------------------------

def load_triplet_arrays(
    science_path: str | Path,
    template_path: str | Path,
    difference_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all three FITS stamps and return them as raw float32 arrays."""
    science = load_fits(science_path)
    template = load_fits(template_path)
    difference = load_fits(difference_path)
    return science, template, difference


def prepare_triplet(
    science: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray,
) -> np.ndarray:
    """
    Normalise each channel independently and stack into (3, 63, 63) float32.

    This is the canonical preprocessing step defined in spec §5.
    """
    channels = [normalize_image(img) for img in (science, template, difference)]
    return np.stack(channels, axis=0).astype(np.float32)  # (3, H, W)


def prepare_triplet_tensor(
    science: np.ndarray,
    template: np.ndarray,
    difference: np.ndarray,
):
    """
    Same as prepare_triplet but returns a float32 torch Tensor of shape (3, 63, 63).
    """
    torch = _get_torch()
    arr = prepare_triplet(science, template, difference)
    return torch.tensor(arr, dtype=torch.float32)
