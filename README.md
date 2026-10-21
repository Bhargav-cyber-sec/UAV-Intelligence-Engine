# UAV Intelligence Engine

An offline, air-gapped multimodal computer vision and intelligence pipeline optimized for Unmanned Aerial Vehicle (UAV) and drone reconnaissance footage.

---

## 📌 Project Overview

The **UAV Intelligence Engine** is engineered for high-altitude aerial surveillance, vehicle tracking, small-object detection, and multi-modal intelligence extraction. Designed from the ground up to operate reliably on air-gapped, offline edge devices without requiring internet connectivity.

### Key Capabilities
- **Aerial Object Detection & Tracking:** Fine-tuned YOLOv11s model trained on the VisDrone aerial benchmark, delivering high-precision vehicle and target localization from top-down angles.
- **ByteTrack with Ghost Reconciliation:** Mitigates track fragmentations caused by occlusions, camera rotations, and high-speed banking maneuvers.
- **Noise-Floor Filtering:** Eliminates spurious single-frame detection flickers (`min_track_frames >= 3`).
- **SAHI Sliced Inference:** Optional sliced inference for ultra-high-resolution aerial frames (1080p/4K) to detect small distant objects.
- **3-Keyframe Extraction:** Extracts start, peak-confidence, and end keyframes per tracked object to provide temporal context for downstream reasoning.
- **Automated Evidence Grid Visualization:** Generates multi-frame synchronized tracking grids with track IDs, bounding boxes, and confidence metrics.

---

## 🎯 Model & Transfer Learning Performance

The pipeline includes both base foundation models and our custom **VisDrone-transferred model**:

| Model | Weights File | Architecture | Primary Specialization | Key Metrics |
|---|---|---|---|---|
| **VisDrone Fine-Tuned (Active)** | `models/yolo/yolo11s_visdrone.pt` | YOLOv11s (9.5M params) | Aerial vehicles (cars, vans, buses, trucks, pedestrians) from drone altitudes | **Car mAP50: 0.783**, Precision: 0.687, Recall: 0.787 |
| **YOLOv11s Foundation** | `models/yolo/yolo11s.pt` | YOLOv11s (9.5M params) | General COCO classes + Aerial aircraft / formations | High confidence on aircraft / formation flight |
| **YOLOv11n Baseline** | `models/yolo/yolo11n.pt` | YOLOv11n (2.6M params) | Lightweight edge fallback | Low VRAM usage |

### VisDrone Transfer Learning Details
- **Dataset:** VisDrone 2019 (6,471 training images, 548 validation images, ~380k annotated instances).
- **Optimization:** AdamW (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`), 50 epochs on NVIDIA RTX 4060 GPU.
- **Aerial Augmentations:** Mosaic (`1.0`), copy-paste (`0.3`), scale jitter (`0.5`), UAV yaw rotation degrees (`10.0`), strictly non-vertical flips.

---

## 📂 Repository Structure

```
UAV_Intelligence_Engine/
├── config/                         # Configuration YAMLs
│   ├── ontology.yaml               # Detection thresholds, SAHI settings, tracking rules
│   ├── paths.yaml                  # Relative paths for models, inputs, and outputs
│   └── uav_visdrone_dataset.yaml   # VisDrone 10-class dataset definition
├── src/                            # Core production modules
│   ├── config.py                   # Central YAML configuration parser
│   ├── media/                      # FFprobe probing and FFmpeg frame sampling
│   ├── vision/                     # YOLO detection, ByteTrack tracking, SAHI, keyframes
│   ├── fusion/                     # Multimodal correlation (vision + audio + telemetry)
│   ├── audio/                      # Audio ingestion and transcription interfaces
│   ├── llm/                        # Local LLM reasoning and summarization
│   ├── tts/                        # Offline Text-To-Speech generation
│   ├── ui/                         # Mission review UI
│   └── report/                     # Automated evidence PDF generation
├── scripts/                        # Production & utility scripts
│   ├── detect_and_grid.py          # Video detection & multi-frame grid generator
│   ├── train_visdrone.py           # VisDrone transfer learning training pipeline
│   ├── validate_model.py           # Model benchmark & side-by-side comparison
│   ├── run_pipeline.py             # Full end-to-end mission pipeline
│   ├── run_sample_pipeline.py      # Quick verification runner on sample footage
│   ├── benchmark_videos.py         # Speed & throughput benchmarking
│   ├── analyze_tracks.py           # Track stability and ID continuity auditor
│   └── verify_hashes.ps1           # SHA-256 weight integrity checker
├── models/                         # Bundled model weights & checksums
│   └── yolo/
│       ├── yolo11s_visdrone.pt     # Fine-tuned aerial weights (18.3 MB)
│       ├── yolo11s.pt              # Base foundation weights (18.4 MB)
│       ├── yolo11n.pt              # Base lightweight weights (5.6 MB)
│       └── hashes.sha256           # Cryptographic SHA-256 signatures
├── data/
│   ├── input/                      # Input video footage (sample.mp4 included)
│   └── output/                     # Generated detection grids, evidence packages, JSONs
├── docs/                           # Architecture, design review, and user guides
│   ├── Documentation.pdf           # Original mission specification
│   ├── design_review.md            # Technical design review & 10-week roadmap
│   ├── uav_yolo_transfer_learning.md # Transfer learning documentation
│   └── OFFLINE_DEVICE_GUIDE.md     # Offline facility PC deployment manual
├── tests/                          # 28 automated pytest unit and integration tests
├── requirements.txt                # Python package dependencies
├── run_pipeline.bat                # One-click execution launcher
├── train_visdrone.bat              # One-click VisDrone training launcher
└── verify_pipeline.bat             # Automated test suite runner
```

---

## 🚀 Quick Start

### 1. Environment Setup
Install dependencies via pip:
```bash
py -3.10 -m pip install -r requirements.txt
```

*(For fully offline/air-gapped facility PCs, use the pre-downloaded wheelhouse in `Environment_and_Installers/`)*.

### 2. Generate Detection Grid on Video
Run YOLO + ByteTrack and generate a multi-frame synchronized tracking grid:
```bash
# Using the VisDrone model for drone/ground vehicle surveillance
py -3.10 scripts/detect_and_grid.py --video data/input/sample.mp4 --model models/yolo/yolo11s_visdrone.pt --step 6 --output data/output/sample_grid.jpg

# Using the COCO model for aircraft / formation flight footage
py -3.10 scripts/detect_and_grid.py --video data/input/airshow.mp4 --model models/yolo/yolo11s.pt --step 6 --output data/output/airshow_grid.jpg
```

### 3. Run the End-to-End Pipeline
```bash
run_pipeline.bat data/input/sample.mp4
```

### 4. Run Automated Test Suite
```bash
verify_pipeline.bat
```
*(All 28 automated tests covering configuration, container probing, keyframe extraction, and tracking algorithms).*

---

## 🛡️ Integrity Verification
All model weights include SHA-256 verification hashes in `models/yolo/hashes.sha256`. Verify weights integrity using:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_hashes.ps1
```

---

## 🔒 Offline Deployment Notice
Offline installers, binary tools (FFmpeg), pre-cached Python wheels, and setup logs are kept in the companion folder:
`Environment_and_Installers/`
This guarantees that this core repository remains lightweight, clean, and directly pushable to GitHub without exceeding upload size thresholds.
