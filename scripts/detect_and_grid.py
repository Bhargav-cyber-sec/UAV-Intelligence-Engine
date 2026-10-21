"""
detect_and_grid.py — Run YOLO + ByteTrack on a video and produce a frame-grid
output image matching the reference format:
  - Grid of sampled frames (every N frames)
  - Colored bounding boxes per unique track ID
  - Labels: "track_id  class  confidence"
  - Frame number in top-left of each cell

Usage:
    py -3.10 scripts/detect_and_grid.py --video path/to/video.mp4 [options]

Options:
    --model         YOLO model weights (default: yolo11s.pt)
    --step          Sample every N frames (default: 6)
    --max_frames    Max frames to include in grid (default: 11)
    --conf          Confidence threshold (default: 0.25)
    --classes       Comma-separated COCO class IDs to filter (default: all)
    --output        Output image path (default: data/output/detection_grid.jpg)
    --grid_cols     Columns in grid (default: 3)
    --imgsz         YOLO inference size (default: 1280)
"""
from __future__ import annotations

import argparse
import colorsys
import random
from pathlib import Path

import cv2
import numpy as np


# ── Colour palette — deterministic per track ID ───────────────────────────────

def track_color(track_id: int) -> tuple[int, int, int]:
    """Return a bright, unique BGR colour for a given track ID."""
    random.seed(track_id * 2654435761 % (2**32))
    h = random.random()
    r, g, b = colorsys.hsv_to_rgb(h, 0.95, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # OpenCV BGR


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_detection(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   track_id: int, class_name: str, conf: float) -> None:
    color = track_color(track_id)
    lw = max(2, int(min(frame.shape[:2]) / 400))  # line width scales with image

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, lw)

    # Label text
    label = f"{track_id}  {class_name}  {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(frame.shape[1], frame.shape[0]) / 1500)
    thickness = max(1, lw - 1)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Background pill for text
    pad = 3
    lx, ly = x1, max(y1 - th - baseline - pad * 2, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + pad * 2, ly + th + baseline + pad * 2), color, -1)
    cv2.putText(frame, label, (lx + pad, ly + th + pad), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_frame_label(frame: np.ndarray, frame_idx: int, highlight: bool = False) -> None:
    label = f"frame {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.5, frame.shape[1] / 1200)
    th = 1
    (tw, fh), bl = cv2.getTextSize(label, font, fs, th)
    color = (0, 255, 255) if highlight else (255, 255, 255)
    # shadow
    cv2.putText(frame, label, (9, fh + 9), font, fs, (0, 0, 0), th + 1, cv2.LINE_AA)
    cv2.putText(frame, label, (8, fh + 8), font, fs, color, th, cv2.LINE_AA)


# ── Grid builder ─────────────────────────────────────────────────────────────

def build_grid(frames_list: list[np.ndarray], cols: int,
               cell_w: int = 640, cell_h: int = 360) -> np.ndarray:
    """Stack annotated frames into a cols-wide grid image."""
    rows = (len(frames_list) + cols - 1) // cols
    # Pad to fill last row
    while len(frames_list) < rows * cols:
        frames_list.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row_imgs = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            cell = frames_list[r * cols + c]
            cell = cv2.resize(cell, (cell_w, cell_h))
            row_cells.append(cell)
        row_imgs.append(np.hstack(row_cells))
    return np.vstack(row_imgs)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] ultralytics not installed. Run: py -3.10 -m pip install ultralytics")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"[ERROR] Video not found: {video_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse class filter
    class_filter = None
    if args.classes:
        class_filter = [int(c.strip()) for c in args.classes.split(",")]

    print(f"\n{'='*60}")
    print(f"  UAV Detection Grid Generator")
    print(f"{'='*60}")
    print(f"  Video  : {video_path}")
    print(f"  Model  : {args.model}")
    print(f"  Step   : every {args.step} frames")
    print(f"  Max    : {args.max_frames} frames in grid")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    # Load model
    model = YOLO(args.model)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total_frames} frames @ {fps:.1f} fps ({total_frames/fps:.0f}s)\n")

    # Collect target frame indices
    target_indices = list(range(0, min(total_frames, args.step * args.max_frames), args.step))
    target_set = set(target_indices)
    print(f"  Sampling frames: {target_indices}\n")

    annotated_cells: list[np.ndarray] = []
    frame_idx = 0
    detection_log = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while frame_idx <= max(target_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set:
            print(f"  → Processing frame {frame_idx}/{total_frames}...", end=" ")

            # Run YOLO tracking (ByteTrack built-in)
            results = model.track(
                frame,
                persist=True,
                conf=args.conf,
                iou=0.5,
                imgsz=args.imgsz,
                tracker="bytetrack.yaml",
                classes=class_filter,
                verbose=False,
            )

            annotated = frame.copy()
            r = results[0]
            n_det = 0

            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    class_name = model.names[cls_id]

                    # Track ID (may be None if tracking lost)
                    tid = int(boxes.id[i]) if boxes.id is not None else (i + 1)

                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    draw_detection(annotated, x1, y1, x2, y2, tid, class_name, conf)
                    detection_log.append({
                        "frame": frame_idx,
                        "track_id": tid,
                        "class": class_name,
                        "conf": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                    })
                    n_det += 1

            draw_frame_label(annotated, frame_idx)
            annotated_cells.append(annotated)
            print(f"{n_det} detections")

        frame_idx += 1

    cap.release()

    # Build grid
    print(f"\n  Building {args.grid_cols}-column grid…")
    grid = build_grid(annotated_cells, cols=args.grid_cols, cell_w=args.cell_w, cell_h=args.cell_h)
    cv2.imwrite(str(output_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  ✓ Grid saved: {output_path}  ({grid.shape[1]}×{grid.shape[0]} px)")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DETECTION SUMMARY")
    print(f"{'='*60}")
    from collections import Counter
    class_counts = Counter(d["class"] for d in detection_log)
    track_ids = set(d["track_id"] for d in detection_log)
    print(f"  Frames analysed : {len(annotated_cells)}")
    print(f"  Unique tracks   : {len(track_ids)}  (IDs: {sorted(track_ids)})")
    print(f"  Total detections: {len(detection_log)}")
    print(f"  By class:")
    for cls, cnt in class_counts.most_common():
        print(f"    {cls:20s}: {cnt}")

    # Save JSON log
    import json
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"summary": {"frames": len(annotated_cells), "unique_tracks": len(track_ids),
                               "total_detections": len(detection_log), "class_counts": dict(class_counts)},
                   "detections": detection_log}, f, indent=2)
    print(f"\n  ✓ JSON log: {json_path}")
    print(f"{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack frame-grid generator")
    parser.add_argument("--video",      required=True,               help="Input video path")
    parser.add_argument("--model",      default="yolo11s.pt",        help="YOLO weights (default: yolo11s.pt)")
    parser.add_argument("--step",       type=int,   default=6,       help="Sample every N frames (default: 6)")
    parser.add_argument("--max_frames", type=int,   default=11,      help="Max frames in grid (default: 11)")
    parser.add_argument("--conf",       type=float, default=0.25,    help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz",      type=int,   default=1280,    help="Inference image size (default: 1280)")
    parser.add_argument("--classes",    default=None,                help="Comma-separated class IDs to keep (e.g. '4,6,8')")
    parser.add_argument("--grid_cols",  type=int,   default=3,       help="Grid columns (default: 3)")
    parser.add_argument("--cell_w",     type=int,   default=640,     help="Grid cell width px (default: 640)")
    parser.add_argument("--cell_h",     type=int,   default=360,     help="Grid cell height px (default: 360)")
    parser.add_argument("--output",     default="data/output/detection_grid.jpg",
                                                                      help="Output image path")
    args = parser.parse_args()
    run(args)
