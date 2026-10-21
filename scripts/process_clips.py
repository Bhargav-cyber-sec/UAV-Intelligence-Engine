"""
process_clips.py — Clip-based video processing pipeline for long UAV videos.

For videos > 2 minutes, this script:
  1. Splits the video into fixed-length clips (default: 60 seconds)
  2. Runs SAHI detection + tracking on each clip independently
  3. Re-links track IDs across clip boundaries via spatial IoU matching
  4. Produces per-clip metrics + an aggregated full-video report

Usage:
  py -3.10 scripts/process_clips.py --video path/to/video.mp4 --output data/output/run1
  py -3.10 scripts/process_clips.py --video path/to/video.mp4 --clip-length 90 --output data/output/run2

Outputs in <output_dir>:
  clips/          raw clip .mp4 files
  clip_N/         per-clip detection grid + metrics JSON
  merged_report.json   full-video aggregated report with re-linked track IDs
  merged_grid.jpg      composite detection grid across all clips
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Video splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_video(video_path: Path, output_dir: Path, clip_length_s: int = 60) -> list[Path]:
    """Split a video into clips of clip_length_s seconds each using FFmpeg.
    Returns list of clip paths in order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_pattern = str(output_dir / "clip_%03d.mp4")

    # Probe video duration
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ],
        capture_output=True, text=True
    )
    duration_s = float(probe.stdout.strip()) if probe.returncode == 0 else 0

    print(f"[split] Video duration: {duration_s:.1f}s → splitting into {clip_length_s}s clips")

    # Split with FFmpeg — segment muxer, re-encode only if needed
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-c", "copy",                        # stream copy — fast, lossless
            "-map", "0",
            "-segment_time", str(clip_length_s),
            "-f", "segment",
            "-reset_timestamps", "1",
            clip_pattern
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[split] FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg split failed")

    clips = sorted(output_dir.glob("clip_*.mp4"))
    print(f"[split] Created {len(clips)} clips")
    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip evaluation (calls evaluate_video.py for each clip)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clip(
    clip_path: Path,
    model_path: Path,
    clip_output_dir: Path,
    clip_index: int,
    conf: float = 0.15,
    imgsz: int = 2048,
    grid_frames: int = 9,
) -> dict:
    """Run evaluate_video.py on a single clip. Returns loaded metrics dict."""
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    grid_out = clip_output_dir / f"clip{clip_index:03d}_grid.jpg"
    json_out = clip_output_dir / f"clip{clip_index:03d}_metrics.json"

    print(f"\n[clip {clip_index:03d}] Evaluating {clip_path.name} ...")

    # Reuse evaluate_video.py as a subprocess so it runs with the same Python env
    script = Path(__file__).parent / "evaluate_video.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--video", str(clip_path),
            "--model", str(model_path),
            "--output-grid", str(grid_out),
            "--output-json", str(json_out),
            "--conf", str(conf),
            "--imgsz", str(imgsz),
            "--grid-frames", str(grid_frames),
        ],
        capture_output=False,  # stream stdout to terminal so user sees progress
    )

    if result.returncode != 0:
        print(f"[clip {clip_index:03d}] WARNING: evaluate_video.py exited with code {result.returncode}")
        return {}

    if json_out.exists():
        with open(json_out) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-clip track re-linking
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def relink_tracks(clip_metrics: list[dict], iou_thresh: float = 0.35) -> dict:
    """
    Re-link track IDs across clip boundaries.

    For each pair of adjacent clips, match end-of-clip-N tracks to start-of-clip-N+1
    tracks by class + spatial IoU. Assign a global_track_id that is stable across clips.

    Returns a merged report dict.
    """
    global_id_counter = 1
    # local_to_global[clip_index][local_id] = global_id
    local_to_global: list[dict[str, int]] = []
    prev_end_tracks: list[dict] = []   # tracks from end of previous clip

    merged_class_counts: dict[str, int] = {}
    total_detections = 0
    total_frames = 0
    global_tracks: list[dict] = []

    for clip_idx, metrics in enumerate(clip_metrics):
        if not metrics:
            local_to_global.append({})
            prev_end_tracks = []
            continue

        # Extract "last known position" of each track from this clip
        clip_tracks = metrics.get("tracks", [])
        clip_start_tracks = []  # tracks that start near the clip boundary
        for t in clip_tracks:
            clip_start_tracks.append(t)

        # Build local→global mapping for this clip
        mapping: dict[str, int] = {}
        matched_prev_ids: set[str] = set()

        if prev_end_tracks and clip_start_tracks:
            for st in clip_start_tracks:
                best_iou = iou_thresh
                best_prev = None
                for pt in prev_end_tracks:
                    if pt.get("class_name") != st.get("class_name"):
                        continue
                    last_box = pt.get("last_bbox") or pt.get("first_bbox")
                    first_box = st.get("first_bbox")
                    if last_box and first_box:
                        score = _iou(last_box, first_box)
                        if score > best_iou:
                            best_iou = score
                            best_prev = pt

                local_id = str(st.get("logical_track_id", st.get("track_id", "")))
                if best_prev is not None:
                    prev_local_id = str(best_prev.get("logical_track_id", best_prev.get("track_id", "")))
                    # Re-use the global ID already assigned to this prev track
                    prev_clip_idx = clip_idx - 1
                    if prev_clip_idx >= 0 and prev_local_id in local_to_global[prev_clip_idx]:
                        mapping[local_id] = local_to_global[prev_clip_idx][prev_local_id]
                        matched_prev_ids.add(prev_local_id)

                if local_id not in mapping:
                    mapping[local_id] = global_id_counter
                    global_id_counter += 1
            # Assign new global IDs to any start tracks that weren't matched
        else:
            for t in clip_start_tracks:
                local_id = str(t.get("logical_track_id", t.get("track_id", "")))
                if local_id not in mapping:
                    mapping[local_id] = global_id_counter
                    global_id_counter += 1

        local_to_global.append(mapping)

        # Update prev_end_tracks for next iteration
        prev_end_tracks = clip_tracks

        # Accumulate global stats
        cc = metrics.get("class_counts", {})
        for cls, cnt in cc.items():
            merged_class_counts[cls] = merged_class_counts.get(cls, 0) + cnt
        total_detections += metrics.get("total_detections", 0)
        total_frames += metrics.get("total_frames", 0)

        for t in clip_tracks:
            local_id = str(t.get("logical_track_id", t.get("track_id", "")))
            global_id = mapping.get(local_id, global_id_counter)
            t_copy = dict(t)
            t_copy["global_track_id"] = global_id
            t_copy["source_clip"] = clip_idx
            global_tracks.append(t_copy)

    # Deduplicate global tracks that are continuations across boundaries
    seen_global_ids: dict[int, dict] = {}
    for t in global_tracks:
        gid = t["global_track_id"]
        if gid not in seen_global_ids:
            seen_global_ids[gid] = t
        else:
            # Merge: extend duration, use max confidence
            existing = seen_global_ids[gid]
            existing["last_frame"] = max(existing.get("last_frame", 0), t.get("last_frame", 0))
            existing["frame_count"] = existing.get("frame_count", 0) + t.get("frame_count", 0)
            existing["max_confidence"] = max(
                existing.get("max_confidence", 0), t.get("max_confidence", 0)
            )

    unique_global_tracks = list(seen_global_ids.values())

    return {
        "total_clips": len(clip_metrics),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "unique_global_tracks": len(unique_global_tracks),
        "class_counts": merged_class_counts,
        "global_tracks": unique_global_tracks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Composite detection grid across clips
# ─────────────────────────────────────────────────────────────────────────────

def build_composite_grid(clip_output_dirs: list[Path], output_path: Path, cols: int = 3) -> None:
    """Stack one representative grid image from each clip into a composite."""
    grids = []
    for d in clip_output_dirs:
        grid_files = sorted(d.glob("*_grid.jpg"))
        if grid_files:
            img = cv2.imread(str(grid_files[0]))
            if img is not None:
                grids.append(img)

    if not grids:
        print("[composite] No grid images found.")
        return

    # Resize all to same width
    target_w = 960
    resized = []
    for g in grids:
        h, w = g.shape[:2]
        scale = target_w / w
        new_h = int(h * scale)
        resized.append(cv2.resize(g, (target_w, new_h)))

    # Stack in rows of `cols`
    rows = []
    for i in range(0, len(resized), cols):
        row_imgs = resized[i:i + cols]
        # Pad row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(row_imgs))

    composite = np.vstack(rows)
    cv2.imwrite(str(output_path), composite)
    print(f"[composite] Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clip-based UAV video detection pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", default=None, help="YOLO model .pt path (defaults to config)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--clip-length", type=int, default=60, help="Clip length in seconds (default: 60)")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=2048, help="YOLO inference image size")
    parser.add_argument("--grid-frames", type=int, default=9, help="Frames per clip in detection grid")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip splitting (assumes clips/ dir already exists in output)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path
    if args.model:
        model_path = Path(args.model)
    else:
        # Try to load from config
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.config import load_config
            cfg = load_config()
            model_path = Path(cfg.paths["models"]["yolo_weights"])
        except Exception:
            # Fallback: search common locations
            candidates = list(Path(".").rglob("yolo11s_visdrone.pt")) + \
                         list(Path(".").rglob("yolo11m_visdrone.pt")) + \
                         list(Path(".").rglob("yolo11n.pt"))
            model_path = candidates[0] if candidates else Path("yolo11s_visdrone.pt")

    print(f"[pipeline] Video   : {video_path}")
    print(f"[pipeline] Model   : {model_path}")
    print(f"[pipeline] Output  : {output_dir}")
    print(f"[pipeline] Clip len: {args.clip_length}s")

    # Step 1: Split video
    if args.no_split:
        clips = sorted(clips_dir.glob("clip_*.mp4"))
        print(f"[pipeline] Skipping split — found {len(clips)} existing clips")
    else:
        clips = split_video(video_path, clips_dir, args.clip_length)

    if not clips:
        print("[pipeline] ERROR: No clips found. Exiting.")
        sys.exit(1)

    # Step 2: Evaluate each clip
    clip_output_dirs: list[Path] = []
    clip_metrics: list[dict] = []

    for i, clip_path in enumerate(clips):
        clip_out = output_dir / f"clip_{i:03d}"
        metrics = evaluate_clip(
            clip_path=clip_path,
            model_path=model_path,
            clip_output_dir=clip_out,
            clip_index=i,
            conf=args.conf,
            imgsz=args.imgsz,
            grid_frames=args.grid_frames,
        )
        clip_metrics.append(metrics)
        clip_output_dirs.append(clip_out)

    # Step 3: Cross-clip track re-linking
    print("\n[pipeline] Re-linking tracks across clip boundaries...")
    merged = relink_tracks(clip_metrics)

    # Add per-clip summaries
    merged["per_clip"] = [
        {
            "clip_index": i,
            "clip_file": str(clips[i].name) if i < len(clips) else "",
            "total_detections": m.get("total_detections", 0),
            "total_frames": m.get("total_frames", 0),
            "class_counts": m.get("class_counts", {}),
            "avg_confidence": m.get("avg_confidence", 0),
        }
        for i, m in enumerate(clip_metrics)
    ]

    # Save merged report
    merged_json = output_dir / "merged_report.json"
    with open(merged_json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[pipeline] Merged report → {merged_json}")

    # Step 4: Composite detection grid
    composite_path = output_dir / "merged_detection_grid.jpg"
    build_composite_grid(clip_output_dirs, composite_path)

    # Step 5: Print summary
    print("\n" + "=" * 60)
    print("  CLIP PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Clips processed     : {merged['total_clips']}")
    print(f"  Total frames        : {merged['total_frames']}")
    print(f"  Total detections    : {merged['total_detections']}")
    print(f"  Unique global tracks: {merged['unique_global_tracks']}")
    print(f"\n  Class breakdown:")
    for cls, cnt in sorted(merged["class_counts"].items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} {cnt}")
    print(f"\n  Per-clip summary:")
    for cp in merged.get("per_clip", []):
        print(f"    clip_{cp['clip_index']:03d}: {cp['total_detections']} detections, "
              f"{cp['total_frames']} frames, conf={cp.get('avg_confidence', 0):.3f}")
    print(f"\n  Reports: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
