# YOLO Live Detection with SORT Tracking

Real-time object detection and tracking using YOLO models with the SORT (Simple Online and Realtime Tracking) algorithm. Supports `.pt`, `.onnx`, and OpenVINO model formats, optimized for CPU inference.

## Features

- **Multi-format support** — PyTorch (`.pt`), ONNX (`.onnx`), and OpenVINO for Intel-optimized inference
- **SORT tracking** — Kalman filter + Hungarian algorithm maintains persistent object IDs across frames
- **Frame-accurate playback** — videos play at their original speed with automatic frame pacing
- **Frame skipping** — on slower hardware, Kalman predictions fill in skipped frames for smooth tracking
- **Video export** — save detection results to `.mp4` for review or presentation
- **Webcam support** — real-time detection from any connected camera

## Setup

### Requirements

- Python 3.10+

### Install Dependencies

```
pip install -r requirements.txt
```

Or install manually:

```
pip install ultralytics onnxruntime opencv-python numpy scipy filterpy
```

For OpenVINO support (Intel CPU optimization):

```
pip install openvino
```

## Usage

### Basic Commands

**Webcam (live camera feed):**
```
python yolo_tracker.py -m best.pt -s 0
```

**Local video file:**
```
python yolo_tracker.py -m best.pt -s video.mp4
```

**Save output to file:**
```
python yolo_tracker.py -m best.pt -s video.mp4 -o output.mp4
```

### Model Formats

**PyTorch (.pt):**
```
python yolo_tracker.py -m best.pt -s video.mp4
```

**ONNX (.onnx):**
```
python yolo_tracker.py -m best.onnx -s video.mp4
```

**OpenVINO (folder):**
```
python yolo_tracker.py -m best_openvino_model_folder -s video.mp4
```

### Detection Settings

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | 0.25 | Confidence threshold (higher = fewer, more accurate detections) |
| `--iou` | 0.45 | NMS IoU threshold |
| `--imgsz` | 640 | Inference resolution (lower = faster, e.g. 416) |

**High confidence detections only:**
```
python yolo_tracker.py -m best.pt -s video.mp4 --conf 0.7
```

**Faster inference with lower resolution:**
```
python yolo_tracker.py -m best.pt -s video.mp4 --imgsz 416
```

### Tracking Settings (SORT)

| Flag | Default | Description |
|------|---------|-------------|
| `--max-age` | 30 | Frames to keep a lost track alive |
| `--min-hits` | 3 | Consecutive detections before showing a track |
| `--sort-iou` | 0.3 | IoU threshold for matching detections to tracks |

### Performance Options

| Flag | Default | Description |
|------|---------|-------------|
| `--skip N` | 0 | Run inference every N+1 frames (Kalman predicts between) |
| `--delay N` | 1 | Display delay in ms (higher = slower playback for inspection) |
| `--display-width` | 1280 | Window width in pixels |

**Optimize for slower CPUs:**
```
python yolo_tracker.py -m best.pt -s video.mp4 --skip 2 --imgsz 416
```

**Slow down playback for frame-by-frame inspection:**
```
python yolo_tracker.py -m best.pt -s video.mp4 --delay 100
```

### All Flags

```
python yolo_tracker.py \
    --model best.pt \
    --source video.mp4 \
    --conf 0.25 \
    --iou 0.45 \
    --imgsz 640 \
    --max-age 30 \
    --min-hits 3 \
    --sort-iou 0.3 \
    --skip 0 \
    --delay 1 \
    --display-width 1280 \
    --output output.mp4
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / Resume |

## Model Conversion

Use `model_converter.ipynb` to convert between formats:

- **PT to ONNX** — same accuracy, faster on CPU
- **PT to OpenVINO** — optimized for Intel CPUs
- **ONNX to OpenVINO** — convert existing ONNX exports

Always keep the `.pt` file as your master copy — it is the only format that can be converted to all others.

## How SORT Tracking Works

1. **Detect** — YOLO detects objects in the current frame
2. **Predict** — Kalman filter predicts where each tracked object should be
3. **Match** — Hungarian algorithm matches new detections to existing tracks using IoU
4. **Update** — Matched tracks update their state; unmatched detections create new tracks
5. **Remove** — Tracks without detections for `max-age` frames are removed

Each tracked object maintains a consistent color across frames. The Kalman filter smoothly predicts positions even on frames where inference is skipped (`--skip`), keeping the visual output fluid.
