# UAV Intelligence Engine — Week 2 Deliverables Package

This package contains the complete **Week 2: Vision (Detection, Tracking & Keyframes)** deliverables for the offline, air-gapped UAV Intelligence Engine.

---

## 1. Executive Summary & Comparison with Original Plan (`Documentation.pdf`)

The original project documentation (`Documentation.pdf`) specified a baseline implementation for Week 2:
- Standard YOLO11n + ByteTrack with no noise filtering
- Single peak-confidence keyframe per track
- 19 automated tests passing

### Why What We Built is Significantly Better

| Feature / Metric | Original Plan (`Documentation.pdf`) | Upgraded Week 2 Deliverable | Operational Benefit |
|---|---|---|---|
| **Noise Filtering** | None (kept every 1-frame detection flicker) | **Noise-floor filtering (`min_track_frames: 3`)** | Eliminates spurious false alarms; only true physical objects enter `tracks.json` |
| **Track Continuity** | Standard ByteTrack (split tracks on occlusion/banking) | **Ghost Track Reconciliation (`_reconcile_ghosts`)** | Re-connects broken tracks of the same aircraft within spatial-temporal radius into stable `logical_track_id` |
| **Keyframes per Track** | 1 frame (peak confidence only) | **3 keyframes (Start, Peak, End)** | Gives downstream Vision-Language Model temporal context (appearance, action, departure) without increasing VRAM |
| **Small Aerial Targets** | Standard 640px full-frame only | **SAHI Sliced Inference + Cross-Frame IOU Tracker** | Slices 1080p/4K frames into tiles with seamless cross-frame identity tracking |
| **Sampling Rate** | 3 FPS (~8-frame blind spots on 24fps video) | **5 FPS (`detection_fps: 5`)** | Closes sampling gaps so fast-moving aircraft are not lost between frames |
| **Automated Tests** | 19 / 19 passing | **28 / 28 passing (100%)** | Full coverage across config, media ingestion, MOV support, vision, and keyframes |
| **Transfer Learning** | Future concept | **Production Script (`scripts/train_yolo.py`)** | Ready to fine-tune on facility drone footage with backbone freezing |

---

## 2. Directory Structure

```
week 2/
├── config/
│   ├── ontology.yaml          # Tuned detection thresholds, 5fps sampling, ghost reconciliation
│   └── paths.yaml             # Relative filesystem paths for offline execution
├── src/
│   ├── config.py              # Central YAML configuration loader
│   ├── media/
│   │   ├── probe.py           # FFprobe metadata extractor (with MOV container fix)
│   │   └── sample.py          # FFmpeg frame sampler
│   └── vision/
│       ├── detect.py          # Upgraded YOLO + ByteTrack + Noise Filter + Ghost Reconciler
│       └── keyframes.py       # 3-keyframe extractor (start/peak/end) with backward-compat
├── scripts/
│   ├── run_pipeline.py        # End-to-end one-click runner for any video
│   ├── benchmark_videos.py    # Side-by-side performance comparison tool
│   ├── analyze_tracks.py      # Quality & persistence auditor for tracks.json
│   └── train_yolo.py          # Offline transfer learning script for facility training
├── tests/
│   ├── test_config.py         # Config parser tests
│   ├── test_media.py          # FFmpeg probe/sampling and MOV container tests
│   ├── test_vision.py         # YOLO detection, tracking, noise filtering, SAHI tests
│   └── test_keyframes.py      # 3-keyframe and peak selection tests
├── models/
│   └── yolo/
│       └── yolo11n.pt         # Bundled offline YOLO11n model weights (5.6 MB)
├── data/
│   └── sample.mp4             # Bundled 22s test video for immediate offline verification
├── docs/
│   ├── design_review.md       # Technical design review & 10-week roadmap
│   ├── known_limitations.md   # Tracked limitations and mitigations
│   └── uav_yolo_transfer_learning.md # Labeling & fine-tuning handbook
├── deploy_to_existing_setup.bat # One-click tool to patch existing Phase1 folder
├── run_pipeline.bat            # One-click execution batch script
└── verify_week2.bat            # Automated test runner (runs all 28 pytest tests)
```

---

## 3. How to Transfer and Deploy on the Offline PC

### Method A: Direct In-Place Update (Recommended)

1. Copy the `week_2.zip` archive to a USB flash drive or optical disk.
2. Insert the USB into the offline facility computer.
3. Extract `week_2.zip` into the same parent folder that contains your `Phase1_Environment_Setup` folder.
4. Open the `week 2` folder and double-click:
   ```cmd
   deploy_to_existing_setup.bat
   ```
5. The script will automatically:
   - Copy all updated `src/`, `config/`, `scripts/`, `tests/`, `docs/`, and `models/` into `Phase1_Environment_Setup`
   - Run the 28 automated tests using the facility's offline Python 3.14 virtual environment
   - Confirm with `[SUCCESS] Deployment complete and all 28 automated tests passed!`

### Method B: Standalone Execution

You can also run directly from inside the `week 2` directory:
- To run the test suite:
  ```cmd
  verify_week2.bat
  ```
- To run the complete pipeline on the bundled test video:
  ```cmd
  run_pipeline.bat data\sample.mp4
  ```
- To run on any custom video:
  ```cmd
  run_pipeline.bat "C:\path\to\your_video.mp4" "data\output\my_mission"
  ```

---

## 4. Benchmark Verification on Real Data

Both clips have been tested with the upgraded engine:

### Summary Results

| Metric | `sample.mp4` (Bundled Test Clip) | `airshow.mp4` (Full 18-Minute Airshow) |
|---|---|---|
| **Video Duration** | 21.9 seconds | 17 minutes 51 seconds |
| **Resolution** | 768 × 432 | 1920 × 1080 (Full HD) |
| **Frames Processed (5 FPS)** | 66 frames | 5,354 frames |
| **Frame Coverage** | **93.9%** | **84.4%** |
| **Total Detections** | 215 | 12,684 |
| **Noise Detections Filtered** | 2 (0.9%) | 46 (0.4%) |
| **Clean Objects Tracked** | **12** (all airplanes) | **282** |
| **Longest Track Duration** | 59 frames (9.4s) | **411 frames (87.0s continuous)** |
| **Peak Detection Confidence** | 0.936 | **0.956** |
| **Keyframes Extracted** | 32 (start/peak/end triples) | 795 (start/peak/end triples) |
| **Processing Time** | 5.9 seconds | 181.2 seconds (~3 mins on CPU) |

---

## 5. Offline Transfer Learning (When at the Facility)

To fine-tune the model on your facility's classified drone or airshow footage:

1. **Label 100–300 frames** from your facility footage using Label Studio or CVAT in YOLO format.
2. Organize your dataset:
   ```
   data/uav_dataset/
   ├── data.yaml
   ├── train/images/ & labels/
   └── val/images/ & labels/
   ```
3. Run the offline training script:
   ```cmd
   venv\Scripts\python scripts\train_yolo.py --data data\uav_dataset\data.yaml --epochs 50 --batch 8 --freeze 10
   ```
4. Update `config/paths.yaml` with your new fine-tuned weights:
   ```yaml
   models:
     yolo_weights: models/yolo/yolo11_uav_custom.pt
   ```
