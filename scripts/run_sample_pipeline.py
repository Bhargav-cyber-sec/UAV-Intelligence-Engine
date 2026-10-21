"""Re-run the full detection pipeline on sample.mp4 with improved detect.py and compare to baseline."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.vision.detect import detect_and_track
from src.vision.keyframes import select_keyframes

frames_dir = Path("data/output/sample_mission/frames")
baseline_dir = Path("data/output/sample_mission/standard")
improved_dir = Path("data/output/sample_mission/improved")

print("Re-running improved detection pipeline on sample.mp4 frames...")
cfg = load_config()
dets, tracks = detect_and_track(frames_dir, improved_dir, cfg=cfg)

# load summary
with open(improved_dir / "detection_summary.json") as f:
    summary = json.load(f)

# load baseline for comparison
with open(baseline_dir / "tracks.json") as f:
    baseline_tracks = json.load(f)

print("\n" + "="*60)
print("  BASELINE (old detect.py)")
print("="*60)
print(f"  Total tracks (including noise):  {len(baseline_tracks)}")
short_baseline = [t for t in baseline_tracks if t["frame_count"] < 3]
print(f"  Noise tracks (< 3 frames):       {len(short_baseline)}")
print(f"  Clean tracks:                    {len(baseline_tracks) - len(short_baseline)}")

print("\n" + "="*60)
print("  IMPROVED (reliability rewrite)")
print("="*60)
print(f"  Total detections:                {summary['total_detections']}")
print(f"  Clean detections:                {summary['clean_detections']}")
print(f"  Noise detections filtered out:   {summary['noise_detections']} ({summary['noise_rate_pct']}%)")
print(f"  Clean tracks:                    {summary['total_tracks']}")
print(f"  Frames with clean detections:    {summary['frames_with_clean_detections']} / {summary['total_frames_processed']}")
print(f"  Ghost window (frames):           {summary['ghost_window_frames']}")
print(f"  Class breakdown:                 {summary['class_counts']}")

print("\n" + "="*60)
print("  CLEAN TRACKS DETAIL")
print("="*60)
for t in tracks:
    print(f"  Track#{t.logical_track_id:3d}  {t.class_name:<12}  "
          f"frames={t.frame_count:3d}  dur={t.duration_s:.1f}s  "
          f"conf_mean={t.mean_confidence:.3f}  conf_max={t.max_confidence:.3f}  "
          f"area={t.area_px_mean:.0f}px²")

# 3-keyframe extraction
print("\nExtracting 3-per-track keyframes (start/peak/end)...")
kfs = select_keyframes(frames_dir, improved_dir / "detections.json", improved_dir)
from collections import Counter
roles = Counter(k.role for k in kfs)
print(f"  Total keyframes: {len(kfs)}  (start={roles['start']}, peak={roles['peak']}, end={roles['end']})")
print(f"  Keyframes written to: {improved_dir / 'keyframes'}")
