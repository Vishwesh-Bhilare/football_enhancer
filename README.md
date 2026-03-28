Download the YOLOv8 segmentation model:

wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt

# ⚽ Football Player Removal Pipeline

A video processing pipeline that detects, tracks, and removes selected players from football broadcast footage — replacing them with a reconstructed background using computer vision and optional Stable Diffusion inpainting.

---

## How It Works

```
Input Video → Player Selection (UI) → Batch Rendering → Output Video
```

1. **Select** players to remove using an interactive OpenCV UI
2. **Run** the rendering pipeline to process the full video
3. **Get** an output video with those players cleanly removed

---

## Features

- **YOLOv8 segmentation** — precise per-frame player detection and masking
- **IoU-based tracking** — consistent player IDs across frames
- **SAM (Segment Anything Model)** — optional mask refinement for cleaner silhouettes
- **Motion-compensated background reconstruction** — homography-aligned frame buffer fills removed regions
- **Stable Diffusion inpainting** — optional high-quality fill using ControlNet (requires CUDA)
- **Temporal smoothing** — reduces flicker between frames
- **Interactive selection UI** — click to select/deselect players before rendering

---

## Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `opencv-python-headless`
- `torch` + `torchvision`
- `ultralytics` (YOLOv8)
- `segment-anything`
- `diffusers`, `transformers`, `accelerate`, `xformers` (for SD inpainting)
- `numpy`, `scipy`, `pillow`

**Download the YOLOv8 segmentation model:**

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt
```

**Optional — SAM checkpoint** (for improved mask quality):

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

> SD inpainting requires a CUDA GPU. All other components run on CPU.

---

## Quickstart

### Run the full pipeline

```bash
python run_pipeline.py --input match.mp4 --output result.mp4
```

You'll be prompted to choose **normal mode** (interactive selection + render) or **debug mode** (uses existing `selection.json`, renders first 10 frames only).

---

### Step 1 — Player Selection

```bash
python main.py --input match.mp4
```

An OpenCV window opens with the video. Use the controls below to mark which players to remove, then save.

| Control | Action |
|---|---|
| Left click | Select player (mark for removal) |
| Right click | Deselect player |
| `Space` | Pause / play |
| `→` | Step forward one frame |
| `←` | Step back one frame |
| `S` | Save selection to `selection.json` |
| `Q` | Quit |

Selected players are highlighted in **red**; unselected in green.

---

### Step 2 — Render Video

```bash
python batch_render.py --input match.mp4 --output result.mp4
```

Or use the advanced renderer (with motion-compensated reconstruction + SD):

```bash
python render_video.py --input match.mp4 --output result.mp4
```

**Debug mode** (saves PNG frames instead of video):

```bash
python render_video.py --input match.mp4 --debug --max-frames 20
```

---

## Project Structure

```
├── main.py                    # Interactive player selection UI
├── batch_render.py            # Rendering pipeline (homography + SD)
├── render_video.py            # Advanced renderer with full reconstruction pipeline
├── run_pipeline.py            # Orchestrates both steps with mode selection
├── config.py                  # All configurable settings
├── selection.json             # Selected player IDs (auto-generated)
│
├── models/
│   └── detector.py            # YOLOv8 player detection
│
├── processing/
│   ├── tracker.py             # IoU-based player tracker
│   ├── effects.py             # Mask creation, refinement, temporal smoothing
│   ├── sam_refiner.py         # SAM-based mask refinement
│   ├── sd_inpainting.py       # Stable Diffusion + ControlNet inpainting
│   └── opacity.py             # Opacity/distance utilities
│
├── utils/
│   └── visualization.py       # Drawing helpers, overlays, debug views
│
└── tests/
    ├── test_tracker_and_effects.py
    └── test_run_pipeline_debug.py
```

---

## Configuration

Edit `config.py` to tune the pipeline:

```python
YOLO_MODEL_NAME = "yolov8m-seg.pt"   # Detection model
DETECTION_CLASSES = [0]               # 0 = person (COCO)
MIN_MASK_AREA = 50                    # Min pixels to trigger SD inpainting

SD_TARGET_SIZE = 512                  # SD inference resolution
SD_PAD_PX = 48                        # Padding around mask crop for SD
SD_PROMPT = "empty football pitch grass, realistic stadium broadcast..."
```

---

## Running Tests

```bash
pytest tests/
```

---

## Notes

- **No CUDA?** SD inpainting is automatically disabled. The pipeline falls back to OpenCV TELEA inpainting + motion-compensated reconstruction, which works well for static/slow-panning cameras.
- **SAM disabled?** If the SAM checkpoint is not found, the pipeline falls back to YOLO masks automatically.
- Player IDs are consistent within a session but reset between runs. Re-run `main.py` if IDs shift between sessions.
