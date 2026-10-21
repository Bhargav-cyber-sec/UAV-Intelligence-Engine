"""
Validate and compare YOLO models on your UAV video.

Usage:
    py -3.10 scripts/validate_model.py --video data/sample.mp4
    py -3.10 scripts/validate_model.py --video path/to/your_video.mp4 --compare

Outputs a side-by-side console table comparing:
  - Base model (yolo11n.pt / COCO)
  - Fine-tuned model (yolo11_visdrone.pt / VisDrone)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _run_detection(video: Path, weights: Path, output_dir: Path, config_path: Path) -> dict:
    """Run full detect_and_track pipeline, return detection_summary.json."""
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.config import load_config
    from src.media.sample import sample_frames
    from src.vision.detect import detect_and_track

    cfg = load_config(str(config_path))
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Sampling frames from {video.name}…")
    sample_frames(str(video), str(frames_dir), cfg=cfg)

    print(f"  Running detection with {weights.name}…")
    detect_and_track(
        frames_dir=frames_dir,
        output_dir=output_dir,
        cfg=cfg,
        model_path_override=weights,
    )

    summary_file = output_dir / "detection_summary.json"
    if not summary_file.exists():
        return {}
    with open(summary_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Validate and compare YOLO models")
    parser.add_argument("--video", type=str, default="data/sample.mp4", help="Path to video file")
    parser.add_argument("--compare", action="store_true", help="Compare base vs VisDrone model")
    parser.add_argument("--weights", type=str, default="models/yolo/yolo11_visdrone.pt",
                        help="Model weights to validate (default: VisDrone fine-tuned)")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        sys.exit(f"[ERROR] Video not found: {video}")

    config_path = Path("config/ontology.yaml")
    out_base = Path("data/output/validation")

    if args.compare:
        print("\n== Comparing Base (COCO) vs VisDrone fine-tuned model ==\n")
        models = {
            "yolo11n (COCO)":         Path("models/yolo/yolo11n.pt"),
            "yolo11_visdrone (ours)": Path(args.weights),
        }
        results = {}
        for name, weights in models.items():
            if not weights.exists():
                print(f"  [{name}] SKIPPED — weights not found: {weights}")
                continue
            print(f"\n[{name}]")
            out = out_base / weights.stem
            results[name] = _run_detection(video, weights, out, config_path)

        print("\n" + "=" * 70)
        print(f"  Comparison on: {video.name}")
        print("=" * 70)
        header = f"  {'Metric':<30} " + "  ".join(f"{n:<25}" for n in results)
        print(header)
        print("  " + "-" * (len(header) - 2))

        metrics = [
            ("Total detections",      "total_detections"),
            ("Clean detections",      "clean_detections"),
            ("Noise filtered",        "noise_detections"),
            ("Total tracks",          "total_tracks"),
            ("Frames w/ detections",  "frames_with_clean_detections"),
            ("Frames processed",      "total_frames_processed"),
        ]
        for label, key in metrics:
            row = f"  {label:<30} "
            for name in results:
                val = results[name].get(key, "N/A")
                row += f"{str(val):<25}  "
            print(row)

        print("\n  Class breakdown:")
        for name, summary in results.items():
            print(f"\n  [{name}]")
            for cls, count in summary.get("class_counts", {}).items():
                print(f"    {cls:<20}: {count} tracks")

    else:
        weights = Path(args.weights)
        if not weights.exists():
            sys.exit(f"[ERROR] Weights not found: {weights}\nRun train_visdrone.bat first.")
        print(f"\n== Validating {weights.name} on {video.name} ==\n")
        summary = _run_detection(video, weights, out_base / weights.stem, config_path)

        print("\n" + "=" * 60)
        print(f"  Model  : {weights.name}")
        print(f"  Video  : {video.name}")
        print("=" * 60)
        for k, v in summary.items():
            if k != "class_counts":
                print(f"  {k:<35}: {v}")
        print("\n  Per-class tracks:")
        for cls, count in summary.get("class_counts", {}).items():
            print(f"    {cls:<20}: {count}")


if __name__ == "__main__":
    main()
