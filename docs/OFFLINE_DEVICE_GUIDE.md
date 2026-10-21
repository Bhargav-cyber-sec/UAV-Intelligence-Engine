# Offline Facility Execution Guide — Week 2 Vision Engine

This document provides complete, step-by-step instructions for deploying, verifying, and running the Week 2 Vision Engine on the facility's air-gapped (non-internet-connected) Windows workstation.

---

## 1. Prerequisites (What Must Already Exist on the Offline PC)

Before starting, confirm that Week 1 environment setup was completed:
1. Python 3.14 virtual environment exists at:
   ```cmd
   Phase1_Environment_Setup\venv\
   ```
2. Bundled FFmpeg exists at:
   ```cmd
   Phase1_Environment_Setup\tools\ffmpeg\bin\ffmpeg.exe
   Phase1_Environment_Setup\tools\ffmpeg\bin\ffprobe.exe
   ```

---

## 2. Transferring the Deliverables via USB / Disk

1. At home, copy `week_2.zip` (or the unzipped `week 2` folder) to your authorized USB drive or optical disc.
2. Plug the USB into the facility offline PC.
3. Copy `week_2.zip` to your project working directory (the parent folder containing `Phase1_Environment_Setup`).
4. Extract `week_2.zip`. Your folder structure on the offline PC will look like this:
   ```
   C:\YourProjectDirectory\
   ├── Phase1_Environment_Setup\     <-- Existing Week 1 installation
   │   ├── venv\
   │   ├── tools\
   │   └── ...
   └── week 2\                       <-- Newly extracted Week 2 folder
       ├── deploy_to_existing_setup.bat
       ├── run_pipeline.bat
       ├── verify_week2.bat
       ├── OFFLINE_DEVICE_GUIDE.md
       ├── config\
       ├── data\
       ├── docs\
       ├── models\
       ├── scripts\
       ├── src\
       └── tests\
   ```

---

## 3. Step-by-Step Deployment Options

Choose either **Method A** (recommended: updates your main project in-place) or **Method B** (runs standalone from `week 2`).

### Method A: Update In-Place into `Phase1_Environment_Setup` (Recommended)

1. Open the `week 2` folder in Windows File Explorer.
2. Double-click:
   ```cmd
   deploy_to_existing_setup.bat
   ```
   *(Or run it from Command Prompt: `cd "week 2" && deploy_to_existing_setup.bat`)*
3. The script will:
   - Copy all updated `src/`, `config/`, `scripts/`, `tests/`, `docs/`, and `models/` into `Phase1_Environment_Setup`
   - Run the 28 automated tests using the facility's offline Python 3.14 virtual environment
   - Display:
     ```
     ==============================================================================
       [SUCCESS] Deployment complete and all 28 automated tests passed!
     ==============================================================================
     ```

### Method B: Run Standalone inside the `week 2` Folder

If you prefer not to touch `Phase1_Environment_Setup`, `week 2` can run independently. It automatically detects the Python virtual environment and FFmpeg binaries in `..\Phase1_Environment_Setup\`:

1. Open a Command Prompt in the `week 2` directory:
   ```cmd
   cd "C:\YourProjectDirectory\week 2"
   ```
2. Verify all 28 automated tests:
   ```cmd
   verify_week2.bat
   ```

---

## 4. How to Test MP4 and MOV Video Files

The pipeline natively supports both `.mp4` (standard H.264/H.265 containers) and `.mov` (QuickTime containers).

### Test 1: Run the Bundled Offline Test Clip (`sample.mp4`)

A 22-second test video is already bundled inside `week 2\data\sample.mp4`.

**Using the One-Click Batch Script:**
```cmd
run_pipeline.bat data\sample.mp4
```

**Or using Python directly:**
```cmd
..\Phase1_Environment_Setup\venv\Scripts\python scripts\run_pipeline.py data\sample.mp4
```

---

### Test 2: Run Any Custom `.mp4` Video (e.g., Drone or Airshow Footage)

To test a video located anywhere on the offline PC:

**Using the Batch Script:**
```cmd
run_pipeline.bat "D:\UAV_Footage\flight_mission_01.mp4" "data\output\flight_mission_01"
```

**Or using Python directly:**
```cmd
..\Phase1_Environment_Setup\venv\Scripts\python scripts\run_pipeline.py "D:\UAV_Footage\flight_mission_01.mp4" "data\output\flight_mission_01"
```

---

### Test 3: Run `.mov` QuickTime Video Files

QuickTime `.mov` files often have variable framerates or missing duration headers in `format.duration`. Our updated `src/media/probe.py` automatically resolves stream-level timestamps for `.mov` containers.

**Using the Batch Script:**
```cmd
run_pipeline.bat "D:\UAV_Footage\aerial_survey.mov" "data\output\aerial_survey_run"
```

**Or using Python directly:**
```cmd
..\Phase1_Environment_Setup\venv\Scripts\python scripts\run_pipeline.py "D:\UAV_Footage\aerial_survey.mov" "data\output\aerial_survey_run"
```

---

## 5. What the Pipeline Does & Where to See the Results

Every video run executes 4 sequential stages and writes its output to `data\output\<mission_name>\`:

```
data\output\<mission_name>\
├── probe.json                    # Stage 1: Video resolution, fps, duration, audio streams
├── frames\                       # Stage 2: Extracted 5 FPS frame images (frame_000001.jpg, ...)
└── vision\
    ├── detections.json           # Stage 3: Raw bounding boxes, scores, and noise flags
    ├── tracks.json               # Stage 3: Reconciled physical objects (duration, conf, trajectory)
    ├── detection_summary.json    # Stage 3: Overall coverage %, noise rate, and class counts
    ├── keyframes.json            # Stage 4: Manifest linking tracks to start/peak/end keyframes
    └── keyframes\                # Stage 4: Actual JPEG keyframe images
        ├── track_0001_airplane_start_frame_000001.jpg
        ├── track_0001_airplane_peak_frame_000024.jpg
        └── track_0001_airplane_end_frame_000059.jpg
```

### How to Inspect and Verify Your Results

1. **View the Keyframe Images:**
   Open `data\output\<mission_name>\vision\keyframes\` in Windows File Explorer. You will see 3 clear images per detected object:
   - `_start_`: The moment the vehicle/aircraft entered the field of view.
   - `_peak_`: The frame with the highest detection confidence (best angle).
   - `_end_`: The moment before it departed or became occluded.

2. **Check the Detection Summary:**
   Open `data\output\<mission_name>\vision\detection_summary.json` with Notepad to see key statistics:
   ```json
   {
     "total_detections": 433,
     "clean_detections": 431,
     "noise_detections": 2,
     "noise_rate_pct": 0.5,
     "total_tracks": 11,
     "frames_with_clean_detections": 103,
     "total_frames_processed": 110,
     "class_counts": {
       "airplane": 11
     }
   }
   ```

3. **Check Track Durations and Confidences:**
   Run the analysis script to print a formatted quality breakdown in Command Prompt:
   ```cmd
   ..\Phase1_Environment_Setup\venv\Scripts\python scripts\analyze_tracks.py
   ```

---

## 6. How to Run Side-by-Side Video Benchmarks

If you have multiple test videos on the offline PC and want to compare detection coverage, tracking continuity, and noise rates side-by-side:

```cmd
..\Phase1_Environment_Setup\venv\Scripts\python scripts\benchmark_videos.py
```

This will output a comparative table:
```
==============================================================
  SIDE-BY-SIDE COMPARISON
==============================================================
  Metric                          sample.mp4          airshow.mp4       
  ------------------------------  ------------------  ------------------
  Frames processed                66                  5354              
  Frame coverage %                93.9%               84.4%             
  Clean tracks                    12                  282               
  Noise filtered %                0.9%                0.4%              
  High-conf tracks (≥0.80)        7                   43                
  Longest track (frames)          59                  411               
  Longest track (seconds)         9.4s                87.0s             
  Peak confidence seen            0.936               0.956             
  Keyframes extracted             32                  795               
```

---

## 7. Troubleshooting Common Offline Issues

| Issue / Error | Cause | Quick Fix |
|---|---|---|
| `'python' is not recognized` | Virtualenv not activated or on PATH | Use the relative path to venv: `..\Phase1_Environment_Setup\venv\Scripts\python.exe` or run via `run_pipeline.bat` |
| `Bundled ffprobe not found` | FFmpeg binaries not found | Ensure `Phase1_Environment_Setup\tools\ffmpeg\bin\` contains `ffmpeg.exe` and `ffprobe.exe` |
| `QuickTime MOV duration is N/A` | Camera wrote duration in stream header instead of container | Fixed in `src/media/probe.py` — it automatically reads stream duration |
| `Detection confidence low (< 0.50)` | General COCO weights detecting distant aerial targets | Expected for small distant specks before transfer learning. Run `scripts/train_yolo.py` on facility labels |
| `Target lost during fast turn` | High angular velocity across frames | In `config/ontology.yaml`, increase `track_continuity.ghost_window_s: 3.0` and `max_motion_radius_px: 200` |
