"""
Media inspection - wraps the bundled FFprobe binary (tools/ffmpeg/bin/ffprobe.exe,
never system PATH) to report the facts every later phase needs: duration,
resolution, fps, codecs, and whether audio is present at all.

Nothing here touches YOLO or Whisper - this module's only job is answering
"what is this file, and does it have audio", which src/vision (Week 2) and
src/audio (Week 4) both depend on before their own work can start. This is
tested against synthetic ffmpeg-generated clips (see tests/test_media.py),
not real UAV footage - per docs/known_limitations.md, that isn't available
yet, and this module's contract doesn't depend on what's actually in frame.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import Config, load_config


class ProbeError(RuntimeError):
    """Raised when ffprobe fails, the file is missing, or no video stream is found."""


@dataclass
class MediaInfo:
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "video_codec": self.video_codec,
            "has_audio": self.has_audio,
            "audio_codec": self.audio_codec,
            "audio_sample_rate": self.audio_sample_rate,
            "audio_channels": self.audio_channels,
        }


def _parse_fps(rate_str: str) -> float:
    """ffprobe reports frame rate as a fraction string like '30000/1001'."""
    if "/" in rate_str:
        num, den = rate_str.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate_str)


def probe(video_path: str | Path, cfg: Config | None = None) -> MediaInfo:
    """Run ffprobe on a media file and return structured info.

    Raises ProbeError if the file doesn't exist, the bundled ffprobe binary
    is missing, ffprobe itself fails, or no video stream is found (an
    audio-only file isn't a valid mission input).
    """
    cfg = cfg or load_config()
    video_path = Path(video_path)
    if not video_path.exists():
        raise ProbeError(f"File not found: {video_path}")

    ffprobe_exe = cfg.ffprobe_exe
    if not ffprobe_exe.exists():
        raise ProbeError(
            f"Bundled ffprobe not found at {ffprobe_exe} - "
            "was tools/ffmpeg/bin included in the transfer package?"
        )

    cmd = [
        str(ffprobe_exe),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed on {video_path}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ProbeError(f"No video stream found in {video_path}")

    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration_val = fmt.get("duration") or video_stream.get("duration") or "0.0"
    if isinstance(duration_val, str) and (duration_val.strip().upper() == "N/A" or not duration_val.strip()):
        duration_val = "0.0"

    return MediaInfo(
        filename=video_path.name,
        duration=float(duration_val),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=_parse_fps(video_stream.get("r_frame_rate", "0/1")),
        video_codec=video_stream.get("codec_name", "unknown"),
        has_audio=audio_stream is not None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        audio_sample_rate=(
            int(audio_stream["sample_rate"])
            if audio_stream and "sample_rate" in audio_stream
            else None
        ),
        audio_channels=(
            int(audio_stream["channels"])
            if audio_stream and "channels" in audio_stream
            else None
        ),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.media.probe <video_path>")
        sys.exit(1)

    info = probe(sys.argv[1])
    print(json.dumps(info.to_dict(), indent=2))