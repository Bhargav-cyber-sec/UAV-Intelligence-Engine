"""
auto_annotate.py â€” Extract frames from a UAV video, auto-label them using the
existing VisDrone model, and produce a ready-to-train YOLO dataset.

This is the "pseudo-labeling" pipeline:
  1. Extract frames at a given FPS from the source video
  2. Run the VisDrone YOLO model to detect objects in each frame
  3. Save detections as YOLO-format .txt label files
  4. Generate a dataset YAML that can be used directly with model.train()
  5. Build a visual QA grid so you can inspect annotation quality

Usage:
  py -3.10 scripts/auto_annotate.py --video path/to/video.mp4 --model models/yolo/yolo11s_visdrone.pt
  py -3.10 scripts/auto_annotate.py --video path/to/video.mp4 --model models/yolo/yolo11s_visdrone.pt --fps 5 --conf 0.20
"""
from __future__ import annotations

import argparse
import random
import shutil
import yaml
from pathlib import Path

import cv2
import numpy as np


VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

CLASS_COLORS = {
    0: (255, 100, 100),   # pedestrian â€” blue
    1: (255, 180, 100),   # people â€” cyan
    2: (100, 255, 100),   # bicycle â€” green
    3: (0, 200, 255),     # car â€” yellow
    4: (0, 140, 255),     # van â€” orange
    5: (0, 80, 255),      # truck â€” red-orange
    6: (255, 0, 180),     # tricycle â€” pink
    7: (200, 0, 255),     # awning-tricycle â€” purple
    8: (0, 255, 200),     # bus â€” teal
    9: (180, 255, 0),     # motor â€” lime
}


def extract_frames(video_path: Path, out_dir: Path, fps: float = 5.0) -> list[Path]:
    """Extract frames from video at specified FPS."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps

    frame_interval = max(1, int(round(video_fps / fps)))
    print(f"[extract] Video: {video_path.name}")
    print(f"          Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"          Duration: {duration_s:.1f}s at {video_fps:.1f} FPS -> {total_frames} total frames")
    print(f"          Extracting every {frame_interval} frames (target {fps} FPS) -> ~{total_frames // frame_interval} frames")

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            stem = f"{video_path.stem}_f{frame_idx:06d}"
            out_path = out_dir / f"{stem}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(out_path)
        frame_idx += 1

    cap.release()
    print(f"[extract] Saved {len(saved)} frames â†’ {out_dir}")
    return saved


def auto_label(
    frame_paths: list[Path],
    labels_dir: Path,
    model_path: Path,
    conf: float = 0.20,
    imgsz: int = 1280,
) -> dict:
    """Run VisDrone model on each frame and write YOLO-format label files."""
    from ultralytics import YOLO

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    stats = {"total_boxes": 0, "class_counts": {}, "frames_with_detections": 0, "empty_frames": 0}

    print(f"\n[label] Auto-annotating {len(frame_paths)} frames with conf>={conf}...")

    for i, frame_path in enumerate(frame_paths):
        results = model.predict(str(frame_path), conf=conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes

        label_path = labels_dir / (frame_path.stem + ".txt")

        if boxes is None or len(boxes) == 0:
            label_path.write_text("")  # empty label file = no objects
            stats["empty_frames"] += 1
            continue

        img_h, img_w = results[0].orig_shape
        lines = []
        for box in boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats["total_boxes"] += 1
            cls_name = VISDRONE_CLASSES.get(cls_id, str(cls_id))
            stats["class_counts"][cls_name] = stats["class_counts"].get(cls_name, 0) + 1

        label_path.write_text("\n".join(lines) + "\n")
        stats["frames_with_detections"] += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(frame_paths):
            print(f"  [{i+1}/{len(frame_paths)}] {stats['total_boxes']} boxes so far...")

    print(f"[label] Done. {stats['total_boxes']} boxes across {stats['frames_with_detections']} frames "
          f"({stats['empty_frames']} empty)")
    print(f"[label] Class breakdown: {stats['class_counts']}")
    return stats


def build_qa_grid(
    frame_paths: list[Path],
    labels_dir: Path,
    output_path: Path,
    n_samples: int = 12,
    cols: int = 4,
) -> None:
    """Build a visual QA grid showing auto-labeled frames with bounding boxes."""
    sample = random.sample(frame_paths, min(n_samples, len(frame_paths)))
    cells = []

    for fp in sample:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Resize to thumbnail
        cell_w, cell_h = 640, 360
        img = cv2.resize(img, (cell_w, cell_h))
        sx, sy = cell_w / w, cell_h / h

        label_path = labels_dir / (fp.stem + ".txt")
        n_boxes = 0
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * cell_w)
                y1 = int((cy - bh / 2) * cell_h)
                x2 = int((cx + bw / 2) * cell_w)
                y2 = int((cy + bh / 2) * cell_h)
                color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cls_name = VISDRONE_CLASSES.get(cls_id, str(cls_id))
                cv2.putText(img, cls_name[0:3], (x1, max(y1 - 2, 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                n_boxes += 1

        label = f"{fp.stem[-10:]}  [{n_boxes} obj]"
        cv2.rectangle(img, (0, 0), (cell_w, 18), (0, 0, 0), -1)
        cv2.putText(img, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cells.append(img)

    if not cells:
        return

    rows_out = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((360, 640, 3), dtype=np.uint8))
        rows_out.append(np.hstack(row))
    grid = np.vstack(rows_out)
    cv2.imwrite(str(output_path), grid)
    print(f"[qa_grid] Saved â†’ {output_path}  ({grid.shape[1]}x{grid.shape[0]})")


def write_dataset_yaml(
    dataset_root: Path,
    train_dir: Path,
    val_dir: Path,
    output_yaml: Path,
    video_name: str,
) -> None:
    """Write a YOLO dataset YAML file for the auto-labeled dataset."""
    data = {
        "path": str(dataset_root),
        "train": str(train_dir.relative_to(dataset_root)),
        "val": str(val_dir.relative_to(dataset_root)),
        "nc": 10,
        "names": VISDRONE_CLASSES,
    }
    output_yaml.write_text(
        f"# Auto-annotated dataset from: {video_name}\n"
        f"# Generated by scripts/auto_annotate.py\n"
        f"# Classes match VisDrone 2019 â€” fine-tune from yolo11s_visdrone.pt weights\n\n"
        + yaml.dump(data, default_flow_style=False, allow_unicode=True)
    )
    print(f"[yaml] Dataset config â†’ {output_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate UAV video frames using VisDrone model")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", default="models/yolo/yolo11s_visdrone.pt", help="YOLO model path")
    parser.add_argument("--output", default="data/custom_dataset", help="Output dataset directory")
    parser.add_argument("--fps", type=float, default=5.0, help="Frame extraction rate (default: 5)")
    parser.add_argument("--conf", type=float, default=0.20,
                        help="Confidence threshold for pseudo-labels (default: 0.20; lower = more boxes but noisier)")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of frames for validation")
    parser.add_argument("--qa-samples", type=int, default=12, help="Number of frames in QA grid")
    args = parser.parse_args()

    video_path = Path(args.video)
    dataset_root = Path(args.output)
    model_path = Path(args.model)

    images_dir = dataset_root / "images" / "all"
    labels_dir = dataset_root / "labels" / "all"

    print("\n" + "=" * 60)
    print("  AUTO-ANNOTATION PIPELINE â€” UAV Intelligence Engine")
    print("=" * 60)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {model_path}")
    print(f"  Output  : {dataset_root}")
    print(f"  FPS     : {args.fps}")
    print(f"  Conf    : {args.conf}")
    print("=" * 60)

    # Step 1: Extract frames
    frame_paths = extract_frames(video_path, images_dir, fps=args.fps)

    if not frame_paths:
        print("[ERROR] No frames extracted. Check video path.")
        return

    # Step 2: Auto-label
    stats = auto_label(frame_paths, labels_dir, model_path, conf=args.conf, imgsz=args.imgsz)

    # Step 3: Train/val split
    random.seed(42)
    random.shuffle(frame_paths)
    n_val = max(1, int(len(frame_paths) * args.val_split))
    val_frames = frame_paths[:n_val]
    train_frames = frame_paths[n_val:]

    train_img_dir = dataset_root / "images" / "train"
    val_img_dir = dataset_root / "images" / "val"
    train_lbl_dir = dataset_root / "labels" / "train"
    val_lbl_dir = dataset_root / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for fp in train_frames:
        shutil.copy(fp, train_img_dir / fp.name)
        lp = labels_dir / (fp.stem + ".txt")
        if lp.exists():
            shutil.copy(lp, train_lbl_dir / lp.name)

    for fp in val_frames:
        shutil.copy(fp, val_img_dir / fp.name)
        lp = labels_dir / (fp.stem + ".txt")
        if lp.exists():
            shutil.copy(lp, val_lbl_dir / lp.name)

    print(f"\n[split] Train: {len(train_frames)} frames | Val: {len(val_frames)} frames")

    # Step 4: Dataset YAML
    yaml_path = dataset_root / "dataset.yaml"
    write_dataset_yaml(dataset_root.resolve(), train_img_dir.resolve(), val_img_dir.resolve(),
                       yaml_path, video_path.name)

    # Step 5: QA grid
    qa_path = dataset_root / "qa_grid.jpg"
    build_qa_grid(frame_paths, labels_dir, qa_path, n_samples=args.qa_samples)

    # Summary
    print("\n" + "=" * 60)
    print("  AUTO-ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"  Frames extracted   : {len(frame_paths)}")
    print(f"  Train / Val        : {len(train_frames)} / {len(val_frames)}")
    print(f"  Total boxes        : {stats['total_boxes']}")
    print(f"  Frames with objects: {stats['frames_with_detections']}")
    print(f"  Empty frames       : {stats['empty_frames']}")
    print(f"\n  Class counts:")
    for cls, cnt in sorted(stats['class_counts'].items(), key=lambda x: -x[1]):
        print(f"    {cls:<20} {cnt}")
    print(f"\n  QA grid            : {qa_path}")
    print(f"  Dataset YAML       : {yaml_path}")
    print(f"\n  Next step â€” fine-tune from VisDrone weights:")
    print(f"    py -3.10 scripts/finetune_custom.py --data {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
