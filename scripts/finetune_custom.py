"""
finetune_custom.py — Fine-tune a VisDrone-pretrained YOLO model on a custom
auto-annotated UAV video dataset.

Key design decisions for a small dataset (~100-200 frames):
  - Start from VisDrone weights (not COCO, not scratch) — model already knows aerial objects
  - Low learning rate (lr0=0.0001) — small nudge, don't forget existing knowledge
  - Freeze backbone (first 10 layers) — only train detection head on small data
  - Fewer epochs (50) — small dataset overfits fast; stop early
  - Heavy augmentation — compensates for small dataset size
  - patience=20 — auto-stop if val mAP50 stops improving

Usage:
  py -3.10 scripts/finetune_custom.py --data data/custom_dataset/v268/dataset.yaml
  py -3.10 scripts/finetune_custom.py --data data/custom_dataset/v268/dataset.yaml --epochs 50 --base-model models/yolo/yolo11s_visdrone.pt
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def finetune(
    data_yaml: str,
    base_model: str = "models/yolo/yolo11s_visdrone.pt",
    epochs: int = 50,
    imgsz: int = 1280,
    batch: int = 8,
    freeze: int = 10,
    device: str = "0",
    output_name: str = "yolo11s_v268_finetuned.pt",
    workers: int = 4,
    lr: float = 0.0001,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[ERROR] ultralytics not installed. Run: py -3.10 -m pip install ultralytics")

    data_path = Path(data_yaml).resolve()
    base_path = Path(base_model).resolve()

    if not data_path.exists():
        raise SystemExit(f"[ERROR] Dataset YAML not found: {data_path}")
    if not base_path.exists():
        raise SystemExit(f"[ERROR] Base model not found: {base_path}\n"
                         "Run train_visdrone.py first to generate it.")

    print("=" * 65)
    print("  CUSTOM FINE-TUNE — UAV Intelligence Engine")
    print("=" * 65)
    print(f"  Base model : {base_path.name}  (VisDrone pretrained)")
    print(f"  Dataset    : {data_path}")
    print(f"  Epochs     : {epochs}  (small dataset — stop early via patience=20)")
    print(f"  Image size : {imgsz}px")
    print(f"  Batch      : {batch}")
    print(f"  Freeze     : first {freeze} backbone layers")
    print(f"  LR         : {lr}  (small — preserve VisDrone knowledge)")
    print(f"  Device     : {device}")
    print("=" * 65)
    print()
    print("  Strategy: VisDrone weights -> fine-tune head on your footage")
    print("  Expected: higher confidence on YOUR specific scene/altitude/camera")
    print()

    model = YOLO(str(base_path))

    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,           # freeze backbone — only head trains on small data
        device=device,
        workers=workers,
        project="runs/detect",
        name="v268_finetune",
        exist_ok=True,
        # Low learning rate — small nudge from VisDrone knowledge toward your domain
        lr0=lr,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Early stopping — with only 139 frames, watch val mAP50 closely
        patience=20,
        # Augmentation — heavy, to compensate for tiny dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45.0,       # aerial UAV can rotate freely
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        copy_paste=0.5,     # paste small objects — critical for pedestrian recall
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=True,
        plots=True,
    )

    # Copy best weights
    print("\n[3/4] Saving fine-tuned weights...")
    import glob as _glob
    candidates = [
        Path("runs/detect/v268_finetune/weights/best.pt"),
    ]
    found = _glob.glob(str(Path.home() / "Downloads/**/v268_finetune/weights/best.pt"), recursive=True)
    candidates += [Path(p) for p in found]

    best_pt = next((p for p in candidates if p.exists()), None)
    target = Path("models/yolo") / output_name

    if best_pt is None:
        print("[WARNING] best.pt not found. Searched:")
        for p in candidates:
            print(f"  {p}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, target)
    print(f"  Saved -> {target}")

    # SHA-256
    sha256 = compute_sha256(target)
    hash_line = f"SHA256 ({output_name}) = {sha256}"
    hash_file = Path("models/yolo/hashes.sha256")
    with open(hash_file, "a") as f:
        f.write(hash_line + "\n")

    # Final metrics
    print("\n[4/4] Validation metrics on custom dataset:")
    try:
        val_results = model.val(data=str(data_path), imgsz=imgsz)
        print(f"  mAP50     : {val_results.box.map50:.4f}")
        print(f"  mAP50-95  : {val_results.box.map:.4f}")
        print(f"  Precision : {val_results.box.mp:.4f}")
        print(f"  Recall    : {val_results.box.mr:.4f}")
    except Exception as e:
        print(f"  (Validation unavailable: {e})")

    print("\n" + "=" * 65)
    print("  FINE-TUNE COMPLETE")
    print("=" * 65)
    print(f"  Model saved : {target}")
    print(f"  SHA-256     : {sha256[:16]}...")
    print(f"\n  To evaluate on v268.mp4:")
    print(f"    py -3.10 scripts/evaluate_video.py \\")
    print(f"      --video _original_downloads/v268.mp4 \\")
    print(f"      --model {target} \\")
    print(f"      --output-grid data/output/v268_finetuned_grid.jpg \\")
    print(f"      --output-json data/output/v268_finetuned_metrics.json")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VisDrone model on custom UAV dataset")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--base-model", default="models/yolo/yolo11s_visdrone.pt",
                        help="Starting weights (default: yolo11s_visdrone.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50; early stop at patience=20)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size (default: 1280)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--freeze", type=int, default=10, help="Backbone layers to freeze (default: 10)")
    parser.add_argument("--device", default="0", help="Device: '0' for GPU, 'cpu' for CPU")
    parser.add_argument("--output", default="yolo11s_v268_finetuned.pt", help="Output model filename")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate (default: 0.0001)")
    args = parser.parse_args()

    finetune(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        output_name=args.output,
        workers=args.workers,
        lr=args.lr,
    )
