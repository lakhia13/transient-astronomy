# ZTF Transient Stamp Triplets (ALeRCE)

## Dataset Summary

A labeled dataset of image triplets from the
[Zwicky Transient Facility (ZTF)](https://www.ztf.caltech.edu/) sky survey,
downloaded from the [ALeRCE broker](https://alerce.science/) and preprocessed
for deep learning. Each sample is one ZTF detection represented as three
co-registered 63×63 pixel sky cutouts — science, template, and difference —
stacked into a single `(3, 63, 63) float32` array.

The dataset supports two classification tasks:

- **Task 1 — Real/Bogus**: Binary. Is this detection a real astronomical
  event, or an imaging artefact (satellite trail, bad pixel, etc.)?
- **Task 2 — Transient Type**: 5-class. If real, classify the source as
  supernova (SN), variable star (VS), active galactic nucleus (AGN),
  or asteroid.

---

## Dataset Details

### Data Instances

Each instance corresponds to one ZTF *detection* (a single observation of
an object on a single night). It contains:

- A `(3, 63, 63) float32` NumPy array with channels ordered as
  `[science, template, difference]`
- A multiclass label (0–4) and a binary real/bogus label (0 or 1)
- Metadata: object ID (`oid`), candidate ID (`candid`), Modified Julian
  Date (`mjd`), and assigned class

### Data Fields

| Field | Type | Description |
|---|---|---|
| `image` | `float32 (3, 63, 63)` | Stacked, z-score normalised stamp triplet |
| `class_label` | `int` | 5-class label: bogus=0, SN=1, AGN=2, VS=3, asteroid=4 |
| `real_bogus_label` | `int` | Binary label: bogus=0, real=1 |
| `class` | `string` | Human-readable class name |
| `oid` | `string` | ALeRCE object identifier (unique per astronomical source) |
| `candid` | `string` | ZTF candidate identifier (unique per detection) |
| `mjd` | `float` | Modified Julian Date of the observation |

### Data Splits

Splits are assigned **chronologically by MJD** (no shuffle) to prevent
temporal data leakage. The train set contains only the oldest observations;
the test set contains the most recent.

| Split | Samples | MJD range |
|---|---|---|
| train | 84,678 | oldest 70% |
| val | 18,078 | next 15% |
| test | 18,134 | most recent 15% |
| **Total** | **123,353** | 58,270 – 61,127 |

### Class Distribution

| Class | Label | Count | Notes |
|---|---|---|---|
| AGN | 3 | 55,858 | Active galactic nuclei |
| VS | 2 | 29,096 | Variable stars |
| SN | 1 | 28,046 | Supernovae |
| bogus | 0 | 9,335 | Imaging artefacts / false detections |
| asteroid | 4 | 1,018 | Solar system objects |

> Note: class counts reflect detections (observations), not unique objects.
> A single object observed on multiple nights contributes one sample per
> detection.

---

## Dataset Creation

### Source Data

Raw data was downloaded from the
[ALeRCE ZTF broker](https://api.alerce.online/ztf/v1) using the
`stamp_classifier` labels, filtered to objects with classifier confidence
≥ 0.5, ordered by descending probability. Only detections with stamps
available on the ALeRCE AVRO store were included.

The original ZTF survey data is publicly available from
[IPAC/Caltech](https://www.ztf.caltech.edu/).

### Preprocessing

1. Three FITS stamps per detection (science, template, difference) were
   loaded from ALeRCE's AVRO store.
2. Each stamp was centre-cropped or zero-padded to exactly 63×63 pixels.
3. NaN and Inf values were replaced with 0.
4. Each channel was independently z-score normalised
   (float64 intermediate to prevent overflow):
   x_norm = (x - mean(x)) / std(x)
5. The three normalised channels were stacked into a `(3, 63, 63) float32`
   array and saved as a `.npy` file.

### Channel Semantics

| Channel | Index | Description |
|---|---|---|
| Science | 0 | New observation — the image where the transient was detected |
| Template | 1 | Reference image of the same sky region (no transient) |
| Difference | 2 | Science minus template — isolates the transient signal |
