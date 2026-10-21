"""
Frame sampling - extracts frames from source video at the configured
detection_fps (config/ontology.yaml: sampling.detection_fps), letterboxed
to yolo_inference_size, using the bundled FFmpeg binary. This is what
src/vision (Week 2) will run YOLO over instead of every frame of the
original 1920x1080 footage - per docs/design_review.md, the original
resolution stays the source of truth for the final highlight, this is
inference-only.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.config import Config, load_config


class SampleError(RuntimeError):
    """Raised when ffmpeg fails, the source file is missing, or the bundled binary is absent."""


def sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    cfg: Config | None = None,
) -> list[Path]:
    """Extract frames at cfg.detection_fps, letterboxed into a
    yolo_inference_size x yolo_inference_size square (aspect preserved,
    padded), into output_dir.

    Returns the sorted list of written frame paths. Re-running this on the
    same output_dir overwrites in place (ffmpeg -y) rather than appending -
    safe to re-run on the same mission without accumulating stale frames.
    """
    cfg = cfg or load_config()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.exists():
        raise SampleError(f"File not found: {video_path}")

    ffmpeg_exe = cfg.ffmpeg_exe
    if not ffmpeg_exe.exists():
        raise SampleError(
            f"Bundled ffmpeg not found at {ffmpeg_exe} - "
            "was tools/ffmpeg/bin included in the transfer package?"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    size = cfg.yolo_inference_size
    vf = (
        f"fps={cfg.detection_fps},"
        f"scale='min({size},iw)':'min({size},ih)':force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    )

    pattern = output_dir / "frame_%06d.jpg"
    cmd = [
        str(ffmpeg_exe),
        "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-qscale:v", "2",
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SampleError(
            f"ffmpeg frame sampling failed on {video_path}: {result.stderr.strip()[-500:]}"
        )

    return sorted(output_dir.glob("frame_*.jpg"))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m src.media.sample <video_path> <output_dir>")
        sys.exit(1)

    frames = sample_frames(sys.argv[1], sys.argv[2])
    print(f"Wrote {len(frames)} frames to {sys.argv[2]}")