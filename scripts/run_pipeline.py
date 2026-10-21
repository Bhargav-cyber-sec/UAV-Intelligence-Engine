"""
End-to-End UAV Video Intelligence Pipeline (Week 2 Deliverable).

Executes the full chain from raw video to tracked objects and keyframe evidence:
  1. src.media.probe   -> inspects container, resolution, fps, duration, audio streams
  2. src.media.sample  -> extracts frames at configured detection_fps (default 5 fps)
  3. src.vision.detect -> YOLO detection + ByteTrack + noise filtering + ghost reconciliation
  4. src.vision.keyframes -> extracts up to 3 keyframes (start/peak/end) per logical track

Usage:
  python scripts/run_pipeline.py [video_path] [output_dir]

Example:
  python scripts/run_pipeline.py data/sample.mp4 data/output/sample_run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.media.probe import probe
from src.media.sample import sample_frames
from src.vision.detect import detect_and_track
from src.vision.keyframes import select_keyframes



def run_full_pipeline(
    video_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    stem = video_path.stem
    out_dir = Path(output_dir or ROOT_DIR / "data" / "output" / f"{stem}_mission")
    frames_dir = out_dir / "frames"
    vision_dir = out_dir / "vision"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  STARTING UAV INTELLIGENCE PIPELINE: {video_path.name}")
    print("=" * 66)
    t_start = time.perf_counter()

    # ── Stage 1: Probe ────────────────────────────────────────────────────────
    print("\n[Stage 1/4] Probing video container and stream metadata...")
    meta = probe(video_path)
    print(f"  Container: {meta.filename}")
    print(f"  Duration : {meta.duration:.2f}s ({meta.duration/60:.1f} min)")
    print(f"  Resolution: {meta.width}x{meta.height} @ {meta.fps:.2f} fps ({meta.video_codec})")
    print(f"  Audio    : {'Yes (' + str(meta.audio_codec) + ')' if meta.has_audio else 'None'}")

    meta_json = out_dir / "probe.json"
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)

    # ── Stage 2: Frame Sampling ───────────────────────────────────────────────
    cfg = load_config()
    target_fps = cfg.detection_fps
    existing_frames = list(frames_dir.glob("frame_*.jpg"))
    if existing_frames:
        print(f"\n[Stage 2/4] Found {len(existing_frames)} existing sampled frames in {frames_dir}.")
    else:
        print(f"\n[Stage 2/4] Sampling frames at {target_fps} fps (letterboxed to {cfg.yolo_inference_size}px)...")
        t_sample = time.perf_counter()
        written = sample_frames(video_path, frames_dir, cfg=cfg)
        print(f"  Extracted {len(written)} frames in {time.perf_counter() - t_sample:.1f}s.")



    # ── Stage 3: Detection & Tracking ─────────────────────────────────────────
    print(f"\n[Stage 3/4] Running YOLO detection, ByteTrack tracking, and ghost reconciliation...")
    t_det = time.perf_counter()
    dets, tracks = detect_and_track(frames_dir, vision_dir, cfg=cfg)
    det_time = time.perf_counter() - t_det
    print(f"  Processed in {det_time:.1f}s.")

    with open(vision_dir / "detection_summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    print(f"  Total raw detections    : {summary['total_detections']}")
    print(f"  Clean detections        : {summary['clean_detections']}")
    print(f"  Noise detections (cut)  : {summary['noise_detections']} ({summary['noise_rate_pct']}%)")
    print(f"  Clean logical tracks    : {summary['total_tracks']}")
    print(f"  Frame detection coverage: {summary['frames_with_clean_detections']} / {summary['total_frames_processed']}")
    print(f"  Class breakdown         : {summary['class_counts']}")

    # ── Stage 4: Keyframe Selection ───────────────────────────────────────────
    print(f"\n[Stage 4/4] Extracting up to 3 keyframes (start / peak / end) per track...")
    kfs = select_keyframes(frames_dir, vision_dir / "detections.json", vision_dir)
    role_counts = Counter(k.role for k in kfs)
    print(f"  Extracted {len(kfs)} keyframes -> {vision_dir / 'keyframes'}")
    print(f"  Keyframe roles breakdown: start={role_counts['start']}, peak={role_counts['peak']}, end={role_counts['end']}")

    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 66)
    print("  MISSION EVIDENCE SUMMARY")
    print("=" * 66)
    for t in sorted(tracks, key=lambda x: -x.frame_count):
        bar = "█" * min(int(t.mean_confidence * 20), 20)
        print(f"  Track#{t.logical_track_id:3d}  {t.class_name:<12}  "
              f"frames={t.frame_count:3d}  dur={t.duration_s:5.1f}s  "
              f"conf {t.mean_confidence:.3f} {bar}")

    print(f"\n[DONE] Pipeline completed in {total_time:.1f}s.")
    print(f"Artifacts written to: {out_dir}")
    return {
        "metadata": meta.__dict__,
        "summary": summary,
        "tracks": [t.__dict__ for t in tracks],
        "keyframe_count": len(kfs),
        "output_dir": str(out_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UAV Intelligence Pipeline")
    parser.add_argument("video_path", nargs="?", default=str(ROOT_DIR / "data" / "sample.mp4"),
                        help="Path to input video file (defaults to data/sample.mp4)")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Path to output directory (optional)")
    args = parser.parse_args()

    run_full_pipeline(args.video_path, args.output_dir)
