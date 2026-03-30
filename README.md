# Astronomical Transient Detection

A machine learning pipeline for classifying real vs. bogus ZTF astronomical transients and predicting transient type (supernova, variable star, AGN, asteroid). Built on live data from the [ALeRCE broker](https://alerce.science/) with time-based evaluation to prevent temporal leakage.

---

## Overview

The [Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu/) scans the sky every few nights and flags any brightness change as a candidate transient. For every real event there are ~10–20 bogus detections caused by satellite trails, bad pixels, and atmospheric artefacts. This project trains deep learning models on ZTF image triplets (science, template, difference) to:

- **Task 1 — Real/Bogus:** Binary classifier. Is this detection a real astronomical event?
- **Task 2 — Transient Type:** 5-class classifier. If real, is it a supernova (SN), variable star (VS), active galactic nucleus (AGN), asteroid, or other?

The pipeline benchmarks against [`braai`](https://github.com/dmitryduev/braai), the production real/bogus model used by ZTF.

---

## Project Structure

```
transient-astronomy/
│
├── data/
│   ├── raw/                        # Downloaded FITS stamp files
│   │   └── <class>/<oid>/<candid>_<type>.fits
│   ├── processed/                  # Preprocessed .npy triplets
│   │   ├── train/ val/ test/       # Time-split .npy files (3×63×63 float32)
│   │   ├── split_labels.csv        # Labels with split, path, and class columns
│   │   └── stats.json              # Per-channel mean/std from training set
│   └── labels.csv                  # Master labels from ALeRCE (oid, candid, class, mjd, paths)
│
├── src/
│   ├── utils.py                    # FITS loading, z-score normalization, label maps
│   ├── dataset.py                  # TransientDataset, time_split, augmentation, weighted sampler
│   ├── models.py                   # ResNet-18, EfficientNet-B0, ViT-Tiny + build_model()
│   ├── losses.py                   # FocalLoss, WeightedBCELoss, MulticlassFocalLoss
│   ├── train.py                    # Training loop with AMP, early stopping, checkpointing
│   └── evaluate.py                 # AUC, AP, F1, confusion matrix, ROC/PR plots
│
├── scripts/
│   ├── download_alerce.py          # Bulk download labeled stamps from ALeRCE
│   └── preprocess.py               # Convert raw FITS → processed .npy splits
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Stage 4: triplet viz, NaN audit, class distribution
│   ├── 02_model_architecture.ipynb # Stage 6: model analysis, param counts, GPU profiling
│   ├── 03_class_imbalance.ipynb    # Stage 7: focal loss, weighted sampler, class weights
│   ├── 04_training.ipynb           # Stage 8: training loop walkthrough and full run
│   └── 05_evaluation.ipynb         # Stage 9: AUC, ROC, PR, F1, confusion matrix, threshold
│
├── checkpoints/                    # Saved model weights (.pt) and training histories (.json)
├── results/                        # Saved plots and evaluation JSON
├── plans/
│   └── spec.md                     # Full project specification
├── pyproject.toml
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python >= 3.12
- CUDA >= 11.8 (RTX 4060 or equivalent, 8 GB VRAM recommended)
- [`uv`](https://github.com/astral-sh/uv) package manager

## Pipeline: Step-by-Step

### Stage 1–3: Data Acquisition

Download labeled ZTF stamp triplets from the ALeRCE broker. The script fetches object lists by class, then downloads science/template/difference FITS stamps for every detection.

```bash
# Full download at spec targets (~24K objects, ~120K stamp triplets)
python scripts/download_alerce.py --workers 8

# Download a specific class with a custom target
python scripts/download_alerce.py --classes SN --targets 500 --workers 8

# Preview what would be downloaded (no disk writes)
python scripts/download_alerce.py --dry-run
```

Stamps are saved to `data/raw/<class>/<oid>/<candid>_<type>.fits`.
A master `data/labels.csv` is written with columns: `oid`, `candid`, `class`, `mjd`, `science_path`, `template_path`, `difference_path`.

The downloader is resumable — re-running skips already-downloaded stamps. HTTP 429 rate limits are handled automatically with exponential back-off.

**Current dataset (downloaded):**

| Stat                             | Value |
|----------------------------------|---|
| Total triplet rows               | 123,353 |
| AGN (Active Galactic Nuclei)     | 55,858 |
| VS (Variable Stars)              | 29,096 |
| SN (Supernovae)                  | 28,046 |
| bogus (False Detection)          | 9,335 |
| asteroid                         | 1,018 |
| MJD (Modified Julian Date) range | 58,270 – 61,127 |

**Label CSV**

| Column | Description |
|---|---|
| `oid` | Object ID (unique per astronomical source) |
| `candid` | Candidate ID (unique per detection) |
| `class` | Astronimical class label |
| `mjd` | Modified Julian Date of detection |
| `science_path` | Path to science FITS stamp |
| `template_path` | Path to template FITS stamp |
| `difference_path` | Path to difference FITS stamp |
---

### Stage 4: Data Exploration

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

Covers (per spec §4):

| Section | Content |
|---|---|
| Class distribution | Triplet counts per class, real:bogus ratio, log-scale charts |
| Triplet visualisation | One example per class (science / template / difference side-by-side) |
| NaN / Inf audit | Scans every stamp; bar chart of bad pixels by class and stamp type |
| Pixel distributions | Per-channel mean KDE plots; box plots of pixel std (dynamic range proxy) |
| Image quality over time | Pixel std vs MJD — checks for night-to-night variation |
| Temporal coverage | MJD range per class with 70/15/15 split boundaries annotated |
| Artefact inspection | Flags FLAT / SATURATED / BADPIX stamps; shows worst-flagged triplet |
| Detections per object | Histogram of light-curve length |
| Normalisation preview | Side-by-side raw ADU vs z-score histograms |
| Summary | Structured findings report |

---

### Stage 5: Preprocessing

```bash
# Full preprocessing run (all ~123K rows)
python scripts/preprocess.py --workers 8

# Quick smoke-test on 500 rows
python scripts/preprocess.py --limit 500 --workers 4

# Force re-run even if output exists
python scripts/preprocess.py --force
```

The script:
1. Reads `data/labels.csv` and applies a **time-based 70/15/15 split** on MJD (no shuffle — prevents temporal leakage)
2. Loads 3 FITS stamps per row, applies per-channel z-score normalisation (float64 intermediate to prevent overflow)
3. Centre-crops/pads non-standard stamp sizes to exactly 63×63
4. Saves each triplet as `(3, 63, 63) float32` in `data/processed/<split>/<candid>.npy`
5. Writes `data/processed/split_labels.csv` and `data/processed/stats.json`

**Processed splits:**

| Split | .npy files |
|---|---|
| train | 84,678 |
| val | 18,078 |
| test | 18,134 |

**Normalisation formula** (spec §5):

```python
def normalize_image(img: np.ndarray) -> np.ndarray:
    img64 = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = img64.mean(), img64.std()
    return ((img64 - mean) / (std if std > 0 else 1.0)).astype(np.float32)
```

**Augmentation** (training only, applied online in `TransientDataset`):

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=180, p=0.5),        # Astronomy has no preferred orientation
    A.GaussNoise(var_limit=(0.01, 0.05), p=0.3),
])
```

Each .npy file is a single preprocessed image triplet stored as a (3, 63, 63) float32 NumPy array. The three channels correspond to:

1. Science image -- the actual sky observation where the transient was detected
2. Template image -- a reference image of the same sky region (no transient present)
3. Difference image -- science minus template, isolating the transient signal

Each stamp is 63x63 pixels, z-score normalized (NaN/Inf values replaced with 0), and centre-cropped/padded to a uniform size.


---

### Stage 6: Model Architecture

```bash
jupyter lab notebooks/02_model_architecture.ipynb
```

Four model families, all accepting `(B, 3, 63, 63) float32` input:

| Model | Task | Params | Notes |
|---|---|---|---|
| `RealBogusResNet18` | Binary (Task 1) | 11.17 M | Modified stem: 3×3 stride-1, maxpool removed |
| `TransientResNet18` | 5-class (Task 2) | 11.17 M | Same stem, 5-way output head |
| `EfficientNetB0` | RB or multiclass | 4.01 M | timm; adaptive avgpool handles 63×63 natively |
| `ViTTiny` | RB or multiclass | 5.49 M | `img_size=63` → 9 patches + CLS = 10 tokens |

**ResNet-18 stem modification** (critical for 63×63 input):

```python
# Standard ImageNet stem collapses 63×63 → 15×15 before residual blocks
# Modified stem: preserve full spatial resolution through layer1
base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
base.maxpool = nn.Identity()
# Result: 63×63 → 63×63 → 32×32 → 16×16 → 8×8 → 512-d vector
```

**Build a model:**

```python
from src.models import build_model

model = build_model('resnet18', task='real_bogus')        # Task 1
model = build_model('resnet18', task='multiclass')        # Task 2
model = build_model('efficientnet_b0', task='real_bogus')
model = build_model('vit_tiny', task='multiclass')
```

---

### Stage 7: Handling Class Imbalance

```bash
jupyter lab notebooks/03_class_imbalance.ipynb
```

Three strategies implemented in `src/losses.py` and `src/dataset.py`:

**Strategy 1 — Focal Loss** (recommended for real/bogus):

```python
from src.losses import FocalLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
# Down-weights easy bogus examples; focuses on hard misclassifications
```

**Strategy 2 — Weighted Random Sampler** (each batch ~50% real):

```python
from src.dataset import make_weighted_sampler
sampler = make_weighted_sampler(train_ds.labels)
loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
```

**Strategy 3 — Class-weighted Cross-Entropy** (multiclass task):

```python
from src.losses import make_weighted_ce
criterion = make_weighted_ce(train_labels, device)
# Rare classes (asteroid) get highest loss weight
```

---

### Stage 8: Training

```bash
jupyter lab notebooks/04_training.ipynb
```

The training loop in `src/train.py` implements:

| Component | Choice |
|---|---|
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Loss | FocalLoss (alpha=0.25, gamma=2.0) |
| Sampler | WeightedRandomSampler |
| Precision | FP16 AMP (`torch.cuda.amp`) |
| Gradient clipping | 1.0 |
| Early stopping | patience=8 epochs (monitored on val AUC) |
| Batch size | 64 |

**Train from Python:**

```python
from src.models import RealBogusResNet18
from src.losses import FocalLoss
from src.train import train

model = RealBogusResNet18()
history = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=FocalLoss(alpha=0.25, gamma=2.0),
    task='real_bogus',
    epochs=50,
    lr=1e-3,
    device=torch.device('cuda'),
    checkpoint_dir='checkpoints/',
    run_name='resnet18_rb',
    patience=8,
)
```

Best checkpoint saved to `checkpoints/resnet18_rb_best.pt`. Per-epoch metrics in `checkpoints/resnet18_rb_history.json`.

---

### Stage 9: Evaluation

```bash
jupyter lab notebooks/05_evaluation.ipynb
```

Full evaluation suite in `src/evaluate.py`:

| Metric | Purpose |
|---|---|
| AUC-ROC | Primary metric (spec §9); robust to class imbalance |
| Average Precision | Area under PR curve; sensitive to false positives |
| F1 @ threshold 0.5 | Interpretability baseline |
| Confusion matrix | Reveals false positive / false negative failure modes |
| Threshold analysis | Finds operating point for recall ≥ 0.95 |

**Current results (untrained baseline — no checkpoint exists yet):**

> These numbers are from the evaluation notebook run on randomly initialised weights. They verify the pipeline is working end-to-end. Real metrics will appear after running the training notebook.

| Metric | Value | Target after training |
|---|---|---|
| Test AUC-ROC | 0.542 | > 0.95 (braai: ~0.99) |
| Average Precision | 0.916 | — |
| F1 @ threshold=0.5 | 0.000 | — |
| Operating threshold | 0.056 | — |
| F1 @ operating threshold | 0.939 | — |
| Recall @ operating threshold | 0.991 | ≥ 0.95 |
| Precision @ operating threshold | 0.892 | — |

---

## Running the Full Pipeline

```bash
# 1. Download data from ALeRCE
python scripts/download_alerce.py --workers 8

# 2. Preprocess FITS → .npy with time-based splits
python scripts/preprocess.py --workers 8

# 3. (Optional) Explore the data
jupyter lab notebooks/01_data_exploration.ipynb

# 4. (Optional) Inspect model architectures
jupyter lab notebooks/02_model_architecture.ipynb

# 5. (Optional) Review imbalance strategies
jupyter lab notebooks/03_class_imbalance.ipynb

# 6. Train the model
jupyter lab notebooks/04_training.ipynb

# 7. Evaluate on the test set
jupyter lab notebooks/05_evaluation.ipynb
```

---

## Key Implementation Details

### Time-based split (no data leakage)

All splits are done by sorting on `mjd` (Modified Julian Date) and taking contiguous slices. This ensures the model is always evaluated on data from the future relative to training, matching real deployment conditions.

```python
df_sorted = df.sort_values('mjd')
n = len(df_sorted)
train_df = df_sorted.iloc[:int(0.70 * n)]            # oldest 70%
val_df   = df_sorted.iloc[int(0.70 * n):int(0.85 * n)]
test_df  = df_sorted.iloc[int(0.85 * n):]            # most recent 15%
```

### Loading processed data

```python
from src.dataset import build_datasets, make_weighted_sampler
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

df = pd.read_csv('data/labels.csv')
train_ds, val_ds, test_ds = build_datasets(
    df,
    task='real_bogus',            # or 'multiclass'
    processed_dir=Path('data/processed'),
)
sampler = make_weighted_sampler(train_ds.labels)
train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4)
```

### Label encoding

```python
# Multiclass
CLASS_TO_IDX = {'bogus': 0, 'SN': 1, 'VS': 2, 'AGN': 3, 'asteroid': 4}

# Binary real/bogus
REAL_BOGUS_MAP = {'bogus': 0, 'SN': 1, 'VS': 1, 'AGN': 1, 'asteroid': 1}
```

---

## Saved Results

After running all notebooks, `results/` contains:

| File | Description |
|---|---|
| `class_imbalance.png` | Real/bogus ratio and per-class bar charts |
| `focal_loss_curve.png` | Focal weight vs confidence for various gamma |
| `loss_comparison.png` | BCE vs Weighted BCE vs Focal loss curves |
| `weighted_sampler_comparison.png` | Batch balance before/after WeightedRandomSampler |
| `class_weights.png` | Balanced class weights for multiclass CE |
| `batch_balance.png` | Per-batch positive fraction histogram |
| `resnet18_feature_maps.png` | Feature map visualisation at each ResNet-18 stage |
| `param_breakdown.png` | Top-8 modules by parameter count per model |
| `model_params_comparison.png` | All models vs braai baseline (bar chart) |
| `vit_patch_grid.png` | ViT-Tiny patch coverage on 63×63 image |
| `receptive_field.png` | ERF and spatial resolution per ResNet-18 stage |
| `gpu_profile.png` | Throughput and VRAM at BS=64 on RTX 4060 |
| `output_head_distributions.png` | Untrained model output distributions |
| `auc_comparison.png` | Model vs random baseline vs braai |
| `roc_curve.png` | ROC curve on test set |
| `pr_curve.png` | Precision-recall curve on test set |
| `confusion_matrix.png` | Normalised and raw count confusion matrices |
| `threshold_curve.png` | Precision/recall/F1 vs decision threshold |
| `eval_summary_resnet18_rb.json` | All test metrics in JSON |

---

## TODO

### Remaining pipeline stages 

- [ ] **Stage 10 — Live Data Testing**: Download nightly ZTF alert tar.gz from `https://ztf.uw.edu/alerts/public/`, parse `.avro` files with `fastavro`, run inference, cross-reference predictions with the Transient Name Server (TNS)
- [ ] **Stage 11 — Ablation Studies**:
  - [ ] Ablation 1: Input channels (all 3 vs science-only vs difference-only vs science+difference)
  - [ ] Ablation 2: Architecture comparison (ResNet-18 vs EfficientNet-B0 vs ViT-Tiny vs simple 3-layer CNN)
  - [ ] Ablation 3: Imbalance strategy comparison (none / sampler only / focal only / combined)
  - [ ] Ablation 4: Training data size learning curves (10% / 25% / 50% / 100%)
- [ ] **Stage 12 — Baseline Comparison**: Install `braai`, run on test set, produce side-by-side AUC/AP/F1/params comparison table

### Training

- [ ] Run the full training job (`notebooks/04_training.ipynb`) — no trained checkpoint exists yet
- [ ] Train the multiclass transient type classifier (`TransientResNet18`, Task 2)
- [ ] Integrate Weights & Biases (`wandb`) for experiment tracking

### Data

- [ ] Download more `bogus` samples to reach the 15,000 target (currently 9,335)
- [ ] Write `scripts/download_ztf_nightly.py` for nightly ZTF alert ingestion

### Code

- [ ] Write `src/live_inference.py` — nightly inference pipeline (parse `.avro`, normalise, predict, output CSV)
- [ ] Add `fastavro` to dependencies for `.avro` alert parsing
- [ ] Add W&B logging hooks to `src/train.py`
- [ ] Write `scripts/run_ablations.py` — automated ablation runner

### Evaluation

- [ ] Produce multi-model ROC curve comparison (all models on same axes, per spec §12)
- [ ] Failure case analysis notebook — show examples the model gets wrong and discuss why
- [ ] Per-class precision/recall table for Task 2 (multiclass classifier)

