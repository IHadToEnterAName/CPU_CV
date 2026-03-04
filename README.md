# Hybrid Road Object Detector — YOLO26s (v19)

5-class YOLO pipeline: **Person | Vehicle | Animal | Traffic Light | Traffic Sign**

---
Drive for report and the rest of helpful documentation: https://drive.google.com/drive/folders/1_mOUVaRk7LkBLPC7aCfANe99JTNeE8mo?usp=sharing

## 1. Requirements

- Python 3.9+
- GPU recommended (CUDA-compatible), CPU supported
- Kaggle account (API key required for dataset download)

### Install Dependencies

```bash
pip install ultralytics albumentations opencv-python tqdm numpy pyyaml pandas matplotlib kagglehub
```

```bash
pip install -r requirements.txt
```

---

## 2. Reproducing the Training

### Step 1 — Kaggle Credentials

Get your API key from [kaggle.com](https://www.kaggle.com) → Account → API → Create New Token. Set the credentials in Cell 2 of the notebook:

```python
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY']      = 'your_key'
```

### Step 2 — Run the Notebook End-to-End

Run all cells in `training_v19.ipynb` sequentially. The pipeline:

1. **Downloads datasets** from Kaggle (cached in `~/.cache/kagglehub/`)
2. **Parses & merges** five source datasets into a unified YOLO format
3. **Balances classes** to hit per-class targets (see below)
4. **Trains YOLO26s** for 120 epochs (single-phase)
5. **Fine-tunes** on animal class for 40 additional epochs
6. **Evaluates** robustness under synthetic corruptions
7. **Exports** to ONNX for CPU inference

### Step 3 — Outputs

- Best weights: `road_detector/hybrid_v17*/weights/best.pt`
- ONNX export: same path with `.onnx` extension
- Robustness report: `robustness_report/`

---

## 3. Datasets

| Source | Kaggle Slug | Classes Contributed | Sample Limit |
|--------|------------|---------------------|-------------|
| BDD100K | `solesensei/solesensei_bdd100k` | person, vehicle, traffic_light, traffic_sign | 25,000 |
| LISA TLD | `mbornoe/lisa-traffic-light-dataset` | traffic_light, traffic_sign | 3,000 |
| IDD | `manjotpahwa/indian-driving-dataset` | person, vehicle | 3,000 |
| Cat-and-Dog | `tongpython/cat-and-dog` | animal (auto-labeled) | 12,000 |
| Animal Counting | `faysalmiah1721758/animal-counting-dataset` | animal (auto-labeled) | 12,000 |

**Animal auto-labeling:** Cat/Dog and Animal Counting images have no bounding-box annotations. YOLO11n is used to auto-detect animal bounding boxes, which are then mapped to class 2 (animal).

### Class Distribution (Final Dataset)

| Class | Train | Val |
|-------|-------|-----|
| 0 — person | 18,800 | 3,410 |
| 1 — vehicle | 23,000 | 4,600 |
| 2 — animal | 12,751 | 2,552 |
| 3 — traffic_light | 12,753 | 2,552 |
| 4 — traffic_sign | 17,250 | 3,450 |

Validation split: 20% of data. Background images (no annotations) from BDD100K are included (200 train / 30 val) to reduce false positives.



# Class Mapping

## 🎯 Final Output Classes

| ID | Class Name    |
| -- | ------------- |
| 0  | person        |
| 1  | vehicle       |
| 2  | animal        |
| 3  | traffic_light |
| 4  | traffic_sign  |

---

## 🧍 Person → `0`

* `person`
* `rider`
* `pedestrian`

---

## 🚗 Vehicle → `1`

* `car`
* `bus`
* `truck`
* `motorcycle`
* `bicycle`
* `van`
* `suv`
* `pickup`
* `minibus`
* `minivan`
* `ambulance`
* `firetruck`
* `fire truck`

---

## 🐾 Animal → `2`

* `animal`
* `dog`
* `cow`
* `horse`
* `cat`
* `sheep`
* `deer`
* `goat`
* `bird`

---

## 🚦 Traffic Light → `3`

* `traffic light`
* `go`

---

## 🛑 Traffic Sign → `4`

* `traffic sign`
* `stop`
* `warning`
* `yield`
* `obs-str-bar-fallback`
* `billboard`


---

## 4. Data Augmentation

### During Dataset Creation

`AUG_FACTOR = 0` — no pre-generated augmentation copies are created. All augmentation happens at training time.

### During Training (Ultralytics Built-in)

YOLO's built-in dynamic augmentations are active during training with these exact values:

- **Mosaic** (1.0) — combines 4 images into one, disabled in last 20 epochs (`close_mosaic=20`)
- **MixUp** (0.15) — blends two images with their labels
- **Copy-Paste** (0.1) — pastes objects from one image onto another
- **HSV shifts** — hue ±0.015, saturation ±0.4, value ±0.3
- **Geometric** — rotation ±5°, translation 10%, scale ±50%, horizontal flip (0.5)
- **Multi-scale** — enabled (random image size variation per batch)

### Optional Albumentations Pipelines (Defined but Not Used by Default)

Two pipelines are defined for experimentation but are **not active** in the default training flow:

**Light augmentor:**
- Horizontal flip (p=0.5)
- Affine: translate 5%, scale 0.9–1.1, rotate ±5° (p=0.5)
- Random brightness/contrast ±0.2 (p=0.4)
- HSV jitter: hue ±8, sat ±15, val ±15 (p=0.3)

**Heavy augmentor:**
- Horizontal flip (p=0.5)
- Affine: translate 8%, scale 0.85–1.15, rotate ±10° (p=0.5)
- Adverse weather/lighting simulation (fog, rain, night, etc.)

---

## 5. Model & Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLO26s (NMS-free, end-to-end) |
| Image size | 960px |
| Batch size | 64 |
| Epochs | 120 (main) + 40 (animal fine-tune) |
| Optimizer | AdamW |
| Learning rate | 0.002 (cosine decay to ×0.01) |
| Weight decay | 0.001 |
| Warmup | 5 epochs |
| `cls` loss weight | 1.5 |
| Mosaic | 1.0 |
| MixUp | 0.15 |
| Copy-Paste | 0.1 |
| Close mosaic | last 20 epochs |
| Multi-scale | Enabled |
| Patience | 50 (main) / 20 (animal fine-tune) |
| Workers | 12 |
| Seed | 42 |

### Why YOLO26s?

- **NMS-free** — end-to-end inference, no post-processing step
- **47.8 mAP@0.5** on COCO (vs 47.0 for YOLO11s)
- **~87ms** CPU ONNX inference at 640px
- Better small-object detection than YOLO11 variants

---

## 6. Robustness Evaluation

After training, the model is evaluated on validation images under 7 synthetic conditions:

| Condition | Description |
|-----------|-------------|
| clean | No modification |
| fog | Weighted blend with white overlay |
| night | Gamma correction (dark) |
| rain | Simulated rain streaks + blur |
| motion_blur | Horizontal kernel blur |
| noise | Gaussian noise (σ=25) |
| overexposure | Brightness scaling (×1.8) |

Results are saved to `robustness_report/robustness_results.csv` with per-class and macro F1 scores.

---

## 7. Inference

### PyTorch

```python
from ultralytics import YOLO
model = YOLO('path/to/best.pt')
results = model.predict('image.jpg', conf=0.25)
```

### ONNX (CPU)

```python
import onnxruntime as ort
sess = ort.InferenceSession('best.onnx', providers=['CPUExecutionProvider'])
```

### Tracking

ByteTrack is configured as the default tracker with a confidence floor of 0.10.

---

## 8. Project Structure

```
training_v19.ipynb       # Main notebook (run end-to-end)
data.yaml                # Generated dataset config for YOLO
final_hybrid_dataset/    # Generated dataset (images + labels)
robustness_report/       # Evaluation outputs
runs/                    # Ultralytics training outputs
```
