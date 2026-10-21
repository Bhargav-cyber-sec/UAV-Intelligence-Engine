"""
Dual-video benchmark: runs the improved detection pipeline on both
sample.mp4 and airshow.mp4 and prints a side-by-side accuracy report.

Usage:
    venv\\Scripts\\python scripts\\benchmark_videos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.vision.detect import detect_and_track
from src.vision.keyframes import select_keyframes


# ── helpers ──────────────────────────────────────────────────────────────────

def print_section(title: str, width: int = 62) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def run_video(label: str, frames_dir: Path, out_dir: Path, cfg) -> dict:
    """Run detect + keyframe pipeline on pre-sampled frames and return metrics."""
    if not frames_dir.exists() or not list(frames_dir.glob("frame_*.jpg")):
        print(f"  [SKIP] {label}: no frames found in {frames_dir}")
        return {}

    t0 = time.perf_counter()
    dets, tracks = detect_and_track(frames_dir, out_dir, cfg=cfg)
    elapsed = time.perf_counter() - t0

    with open(out_dir / "detection_summary.json") as f:
        summary = json.load(f)

    # keyframes
    kfs = select_keyframes(frames_dir, out_dir / "detections.json", out_dir)
    role_counts = Counter(k.role for k in kfs)

    # per-track confidence buckets
    confs = [t.mean_confidence for t in tracks]
    high    = sum(1 for c in confs if c >= 0.80)
    medium  = sum(1 for c in confs if 0.60 <= c < 0.80)
    low     = sum(1 for c in confs if c < 0.60)

    # longest-running track
    best = max(tracks, key=lambda t: t.frame_count) if tracks else None

    return {
        "label": label,
        "total_frames": summary["total_frames_processed"],
        "frames_covered": summary["frames_with_clean_detections"],
        "coverage_pct": round(100 * summary["frames_with_clean_detections"] /
                              max(summary["total_frames_processed"], 1), 1),
        "total_detections": summary["total_detections"],
        "clean_detections": summary["clean_detections"],
        "noise_detections": summary["noise_detections"],
        "noise_pct": summary["noise_rate_pct"],
        "clean_tracks": summary["total_tracks"],
        "class_counts": summary["class_counts"],
        "conf_high": high,
        "conf_med": medium,
        "conf_low": low,
        "best_track_frames": best.frame_count if best else 0,
        "best_track_dur_s": best.duration_s if best else 0,
        "best_track_conf": best.mean_confidence if best else 0,
        "best_track_max_conf": best.max_confidence if best else 0,
        "keyframes_total": len(kfs),
        "kf_start": role_counts["start"],
        "kf_peak": role_counts["peak"],
        "kf_end": role_counts["end"],
        "elapsed_s": round(elapsed, 1),
        "tracks": tracks,
    }


def print_report(r: dict) -> None:
    if not r:
        return
    label = r["label"]
    print_section(f"VIDEO: {label}")

    print(f"  Frames processed          : {r['total_frames']}")
    print(f"  Frames with detections    : {r['frames_covered']} / {r['total_frames']}  ({r['coverage_pct']}%)")
    print(f"  Processing time           : {r['elapsed_s']}s")

    print()
    print(f"  Total raw detections      : {r['total_detections']}")
    print(f"  Clean detections          : {r['clean_detections']}")
    print(f"  Noise filtered out        : {r['noise_detections']}  ({r['noise_pct']}%)")

    print()
    print(f"  Clean tracks (objects)    : {r['clean_tracks']}")
    print(f"  Class breakdown           : {r['class_counts']}")

    print()
    print("  Track confidence quality:")
    print(f"    High (>= 0.80)          : {r['conf_high']} tracks")
    print(f"    Medium (0.60 – 0.80)    : {r['conf_med']} tracks")
    print(f"    Low (< 0.60)            : {r['conf_low']} tracks")

    print()
    print("  Most persistent track:")
    print(f"    Frames seen             : {r['best_track_frames']}")
    print(f"    Duration                : {r['best_track_dur_s']:.1f}s")
    print(f"    Mean confidence         : {r['best_track_conf']:.3f}")
    print(f"    Peak confidence         : {r['best_track_max_conf']:.3f}")

    print()
    print(f"  Keyframes extracted       : {r['keyframes_total']}  "
          f"(start={r['kf_start']}, peak={r['kf_peak']}, end={r['kf_end']})")

    print()
    print("  All clean tracks:")
    for t in sorted(r["tracks"], key=lambda x: -x.frame_count):
        bar = "█" * min(int(t.mean_confidence * 20), 20)
        print(f"    Track#{t.logical_track_id:4d}  {t.class_name:<14}  "
              f"frames={t.frame_count:4d}  dur={t.duration_s:5.1f}s  "
              f"conf {t.mean_confidence:.3f} {bar}")


def print_comparison(r1: dict, r2: dict) -> None:
    if not r1 or not r2:
        return
    print_section("SIDE-BY-SIDE COMPARISON")

    rows = [
        ("Metric", r1["label"], r2["label"]),
        ("-" * 30, "-" * 18, "-" * 18),
        ("Frames processed", r1["total_frames"], r2["total_frames"]),
        ("Frame coverage %", f"{r1['coverage_pct']}%", f"{r2['coverage_pct']}%"),
        ("Clean tracks", r1["clean_tracks"], r2["clean_tracks"]),
        ("Noise filtered %", f"{r1['noise_pct']}%", f"{r2['noise_pct']}%"),
        ("High-conf tracks (≥0.80)", r1["conf_high"], r2["conf_high"]),
        ("Longest track (frames)", r1["best_track_frames"], r2["best_track_frames"]),
        ("Longest track (seconds)", f"{r1['best_track_dur_s']:.1f}s", f"{r2['best_track_dur_s']:.1f}s"),
        ("Peak confidence seen", f"{r1['best_track_max_conf']:.3f}", f"{r2['best_track_max_conf']:.3f}"),
        ("Keyframes extracted", r1["keyframes_total"], r2["keyframes_total"]),
        ("Processing time", f"{r1['elapsed_s']}s", f"{r2['elapsed_s']}s"),
    ]
    for row in rows:
        print(f"  {str(row[0]):<30}  {str(row[1]):<18}  {str(row[2]):<18}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()

    videos = [
        ("sample.mp4",  Path("data/output/sample_mission/frames"),  Path("data/output/sample_mission/improved")),
        ("airshow.mp4", Path("data/output/airshow_mission/frames"), Path("data/output/airshow_mission/improved")),
    ]

    results = []
    for label, frames_dir, out_dir in videos:
        print(f"\nRunning: {label}  (frames in {frames_dir})")
        r = run_video(label, frames_dir, out_dir, cfg)
        results.append(r)
        print_report(r)

    if len(results) == 2:
        print_comparison(results[0], results[1])

    print("\n✓ Benchmark complete.\n")
