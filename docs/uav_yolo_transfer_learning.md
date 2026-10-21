# UAV & Airshow YOLO Transfer Learning & Small Object Optimization

## 1. Executive Summary & Root Cause Analysis

During Week 2 testing at the facility, standard YOLO11n failed on two specific target clips:
1. **Airshow Footage**: Camera mounted on an aircraft recording distant airborne planes in high-speed flight.
2. **UAV Surveillance Footage**: High-altitude drone footage monitoring buildings, vehicles, and personnel from nadir/oblique overhead angles.

### Why Standard COCO Pretrained YOLO Failed

| Issue | Root Cause | Impact | Fix / Mitigation |
|---|---|---|---|
| **Distant airplanes & vehicles missed** | 640x640 letterboxing shrinks a 1920x1080 frame; a 20x20px distant object becomes < 6x6px at stride 8/16 feature maps | Low confidence (0.3-0.6) or completely missed | **SAHI (Slicing Aided Hyper Inference)** sliding-window tiling |
| **Top-down UAV view failures** | COCO is 100% eye-level ground footage; overhead vehicle roofs & oblique perspectives are Out-Of-Distribution | Tracker loses track identity & misses detections | **Transfer Learning / Fine-Tuning** on VisDrone / custom labeled footage |
| **QuickTime `.mov` file errors** | QuickTime `.mov` containers with Variable Frame Rates (VFR) or non-standard atoms | Frame extraction drops frames or fails in naive readers | FFmpeg container-agnostic demuxing with explicit frame rate control |

---

## 2. Architecture Enhancements (Week 2 Updates)

### A. Slicing Aided Hyper Inference (SAHI) Tiling
We added built-in sliding window tile slicing directly into `src/vision/detect.py`. 
When `sahi.enabled: true` in `config/ontology.yaml`:
- The input frame (1920x1080 or 4K) is sliced into overlapping 640x640 tiles (default 20% overlap).
- Detection runs at **full native resolution** per tile.
- Tile bounding boxes are mapped back to original image coordinates.
- Non-Maximum Suppression (NMS) removes duplicate detections across overlap boundaries.

**Configuration (`config/ontology.yaml`):**
```yaml
detection:
  confidence_threshold: 0.20
  tracker: bytetrack.yaml
  sahi:
    enabled: true
    slice_height: 640
    slice_width: 640
    overlap_ratio: 0.20
```

---

## 3. Transfer Learning & Fine-Tuning Guide (Step-by-Step)

To achieve high confidence (0.80+) and domain accuracy on facility footage:

### Step 1: Frame Extraction & Annotation (At Facility / Offline)
1. Extract representative frames from facility clips using `src.media.sample`.
2. Annotate frames using an offline tool (e.g. **AnyLabeling**, **CVAT**, or **LabelImg**).
3. Export annotations in **YOLO format** (`.txt` files containing `class_id x_center y_center width height` normalized 0..1).

### Step 2: Dataset Directory Layout
Organize the dataset folder at home:
```
uav_dataset/
  ├── data.yaml
  ├── train/
  │   ├── images/
  │   └── labels/
  └── val/
      ├── images/
      └── labels/
```

`data.yaml` example:
```yaml
path: uav_dataset
train: train/images
val: val/images
names:
  0: airplane
  1: uav
  2: car
  3: truck
  4: building
```

### Step 3: Run Fine-Tuning at Home (GPU environment)
Run `scripts/train_yolo.py`:
```bash
py -3.10 scripts/train_yolo.py --data path/to/data.yaml --weights yolo11s.pt --epochs 50 --imgsz 1024 --batch 16 --output yolo11_uav_custom.pt
```

*Tip*: Using `yolo11s.pt` or `yolov8x-visdrone.pt` as base weights provides a strong feature backbone for aerial targets.

### Step 4: Transfer to Facility PC
1. Copy the fine-tuned weights `yolo11_uav_custom.pt` into `models/yolo/`.
2. Generate new SHA256 checksum:
   ```powershell
   Get-FileHash models\yolo\yolo11_uav_custom.pt -Algorithm SHA256
   ```
3. Update `config/paths.yaml`:
   ```yaml
   models:
     yolo_weights: models/yolo/yolo11_uav_custom.pt
   ```
4. Run verification tests:
   ```powershell
   venv\Scripts\python -m pytest
   ```
