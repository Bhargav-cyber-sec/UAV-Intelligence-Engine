"""
VisDrone Transfer Learning Script — UAV Intelligence Engine (Week 2+).

This script does EVERYTHING in one shot:
  1. Downloads the VisDrone 2019 DET dataset automatically via Ultralytics
     (train: 6,471 images | val: 548 images | 10 UAV-specific classes)
  2. Fine-tunes a YOLO model on VisDrone aerial imagery
  3. Saves best weights to models/yolo/yolo11_visdrone.pt
  4. Validates the fine-tuned model and prints a metrics summary
  5. Generates an SHA-256 hash for facility transfer integrity check

VisDrone 10 classes (all aerial/drone footage specific):
  0: pedestrian   1: people       2: bicycle     3: car
  4: van          5: truck        6: tricycle    7: awning-tricycle
  8: bus          9: motor

Optimized for high-altitude drone footage (vehicles, small objects, top-down view):
  - yolo11s.pt base: ~2x better accuracy than nano with modest compute cost
  - imgsz=640: fast training (~2-3h on RTX 4060), good small-object recall
  - freeze=0: full fine-tune (6.4K VisDrone images is enough; better for small objects)
  - copy_paste=0.3: paste extra small objects into scenes (critical for aerial density)
  - Offline deployment friendly: once trained, no internet required

Usage (GPU recommended, ~2-3 hrs on RTX 4060):
    py -3.10 scripts/train_visdrone.py --model yolo11s.pt --epochs 50 --imgsz 640 --batch 8

Higher accuracy (~5-6h, 1280px):
    py -3.10 scripts/train_visdrone.py --model yolo11s.pt --epochs 50 --imgsz 1280 --batch 4

CPU-only fallback (slow, ~6-12 hrs):
    py -3.10 scripts/train_visdrone.py --epochs 30 --imgsz 640 --batch 4 --device cpu

After training, copy the output weights to your facility PC (no internet needed there):
    models/yolo/yolo11_visdrone.pt
Then update config/paths.yaml:
    models:
      yolo_weights: models/yolo/yolo11_visdrone.pt
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def train_visdrone(
    model_weights: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch_size: int = 8,
    freeze_layers: int = 10,
    device: str = "0",   # "0" = first GPU; "cpu" for CPU-only
    output_name: str = "yolo11_visdrone.pt",
    workers: int = 4,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "[ERROR] ultralytics is not installed.\n"
            "Run:  py -3.10 -m pip install ultralytics\n"
        )

    print("=" * 70)
    print("  VisDrone Transfer Learning — UAV Intelligence Engine")
    print("=" * 70)
    print(f"  Base model  : {model_weights}")
    print(f"  Dataset     : VisDrone 2019 (auto-download via Ultralytics)")
    print(f"  Epochs      : {epochs}")
    print(f"  Image size  : {imgsz}px")
    print(f"  Batch size  : {batch_size}")
    print(f"  Freeze (N)  : first {freeze_layers} backbone layers frozen")
    print(f"  Device      : {device}")
    print("=" * 70)

    # ── Load base model ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading base weights: {model_weights}")
    model = YOLO(model_weights)

    # ── Train on VisDrone ────────────────────────────────────────────────────
    print("\n[2/4] Starting VisDrone fine-tuning…")
    print("      (Ultralytics will auto-download VisDrone if not found.)")
    print("      Dataset size: ~6,471 train + 548 val images, ~10 classes.")
    print("      This may take a while on CPU — consider running overnight.\n")

    results = model.train(
        data="VisDrone.yaml",            # built-in Ultralytics dataset config
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        freeze=freeze_layers,            # freeze backbone to preserve features, only train head
        device=device,
        workers=workers,
        project="runs/detect",
        name="visdrone_finetune",
        exist_ok=True,
        # ── Augmentation — tuned for high-altitude small-object aerial footage ──
        hsv_h=0.015,        # slight hue variation (road colour, vegetation, sky)
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45.0,       # strong rotation: aerial view has no fixed "up" (UAV can bank/yaw freely)
        translate=0.1,
        scale=0.5,          # scale jitter: simulates varying altitude — kept at 0.5 (safe with multi_scale off)
        flipud=0.0,         # no vertical flip (sky must stay up in aerial footage)
        fliplr=0.5,
        mosaic=1.0,         # mosaic: 4 images tiled — greatly helps small object recall
        close_mosaic=10,    # disable mosaic in last 10 epochs for stable convergence
        copy_paste=0.5,     # raised from 0.3 → more small object pasting — critical for pedestrian/bicycle recall
        # multi_scale disabled — causes BatchNorm single-value error at imgsz=1280 when tile shrinks to 32px
        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # ── Loss weights — upweight box regression for small aerial objects ───
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # ── Validation ────────────────────────────────────────────────────────
        val=True,
        plots=True,
    )

    # ── Copy best weights ─────────────────────────────────────────────────────
    print("\n[3/4] Saving fine-tuned weights…")
    # Ultralytics saves relative to its resolved project path — check both locations
    cwd = Path.cwd()
    candidate_paths = [
        cwd / "runs/detect/visdrone_finetune/weights/best.pt",
        cwd.parent / "Phase1_Environment_Setup/runs/detect/runs/detect/visdrone_finetune/weights/best.pt",
        cwd.parent.parent / "Phase1_Environment_Setup/Phase1_Environment_Setup/runs/detect/runs/detect/visdrone_finetune/weights/best.pt",
    ]
    # Also glob search under Downloads for safety
    import glob
    found = glob.glob(str(Path.home() / "Downloads/**/visdrone_finetune/weights/best.pt"), recursive=True)
    candidate_paths += [Path(p) for p in found]

    best_pt = next((p for p in candidate_paths if p.exists()), None)
    target  = Path("models/yolo") / output_name

    if best_pt is None:
        print(f"[WARNING] best.pt not found in any expected location. Searched:")
        for p in candidate_paths:
            print(f"  {p}")
        return

    print(f"  Found weights at: {best_pt}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_pt, target)
    print(f"  ✓ Saved to: {target}")

    # ── Hash for integrity check ─────────────────────────────────────────────
    sha256 = compute_sha256(target)
    hash_line = f"SHA256 ({output_name}) = {sha256}"
    print(f"\n[4/4] SHA-256 hash (add to hashes.sha256 before facility transfer):")
    print(f"  {hash_line}")

    hash_file = Path("models/yolo/hashes.sha256")
    with open(hash_file, "a") as f:
        f.write(hash_line + "\n")
    print(f"  ✓ Appended to: {hash_file}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRANSFER LEARNING COMPLETE")
    print("=" * 70)
    print(f"\n  Fine-tuned model : {target}")
    print(f"  SHA-256          : {sha256[:16]}…")
    print("""
  Next steps:
    1. Copy models/yolo/yolo11_visdrone.pt to facility PC.
    2. Update config/paths.yaml:
         models:
           yolo_weights: models/yolo/yolo11_visdrone.pt
    3. Verify transfer integrity:
         py -3.10 scripts/verify_hashes.ps1  (or compare SHA manually)
    4. Run the pipeline on your UAV video:
         run_pipeline.bat path\\to\\your_video.mp4
    5. Validate results — check detection_summary.json for improved counts.
""")

    print("  Validation metrics on VisDrone val set:")
    try:
        val_results = model.val(data="VisDrone.yaml", imgsz=imgsz)
        print(f"  mAP50     : {val_results.box.map50:.4f}")
        print(f"  mAP50-95  : {val_results.box.map:.4f}")
        print(f"  Precision : {val_results.box.mp:.4f}")
        print(f"  Recall    : {val_results.box.mr:.4f}")
    except Exception as e:
        print(f"  (Validation metrics unavailable: {e})")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO on VisDrone aerial dataset for UAV footage"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt",
        help="Base model weights (default: yolo11n.pt). Use yolo11s.pt or yolo11m.pt for more accuracy.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100). Use 30 for a quick CPU test.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Training image size (default: 1280). Use 640 for CPU/low-memory.",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8). Reduce to 4 if GPU OOM with 4GB VRAM.",
    )
    parser.add_argument(
        "--freeze", type=int, default=10,
        help="Number of backbone layers to freeze (default: 10). Set 0 for full fine-tune.",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Training device: '0' (GPU), 'cpu'. Default: '0' (first GPU).",
    )
    parser.add_argument(
        "--output", type=str, default="yolo11_visdrone.pt",
        help="Output model filename saved in models/yolo/ (default: yolo11_visdrone.pt)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Dataloader workers (default: 4). Set to 0 on Windows if errors occur.",
    )
    args = parser.parse_args()

    train_visdrone(
        model_weights=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        freeze_layers=args.freeze,
        device=args.device,
        output_name=args.output,
        workers=args.workers,
    )
