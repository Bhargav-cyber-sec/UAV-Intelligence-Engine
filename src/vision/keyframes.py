"""
Keyframe selection (Week 2 — upgraded to 3-keyframe-per-track).

For every clean (non-noise) logical track, selects THREE representative frames:
  - start  : first frame the track was seen
  - peak   : frame with the highest confidence
  - end    : last frame the track was seen

This directly addresses docs/known_limitations.md #6 ("single peak-frame
keyframe caption can miss what actually mattered in an event") — 3 keyframes
give the VL captioning stage in Week 5 start/peak/end context without blowing
up VRAM cost (they are still individual sequential calls, flat VRAM).

Outputs:
  - <output_dir>/keyframes/<keyframe_file>   one JPEG per selected frame
  - <output_dir>/keyframes.json              manifest of all keyframes
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


class KeyframeError(RuntimeError):
    """Raised when required inputs are missing."""


@dataclass
class Keyframe:
    logical_track_id: int
    class_name: str
    role: str            # 'start' | 'peak' | 'end'
    frame_index: int
    source_frame: str
    keyframe_file: str
    confidence: float
    bbox_xyxy: list[float]

    @property
    def track_id(self) -> int:
        """Alias for backward compatibility with earlier pipeline stages."""
        return self.logical_track_id



def select_keyframes(
    frames_dir: str | Path,
    detections_json: str | Path,
    output_dir: str | Path,
) -> list[Keyframe]:
    """Select up to 3 keyframes per logical track (start / peak / end).

    Reads *detections_json* (written by src.vision.detect) and copies the
    chosen source frames into *output_dir*/keyframes/.  Skips detections
    flagged noise=True.
    """
    frames_dir     = Path(frames_dir)
    detections_json = Path(detections_json)
    output_dir     = Path(output_dir)

    if not detections_json.exists():
        raise KeyframeError(
            f"{detections_json} not found — run src.vision.detect first."
        )

    with open(detections_json, encoding="utf-8") as f:
        raw = json.load(f)

    # Group clean detections by logical_track_id.
    # Fall back to track_id if logical_track_id key absent (backward compat
    # with older detections.json that predate this rewrite).
    by_track: dict[int, list[dict]] = {}
    for det in raw:
        if det.get("noise", False):
            continue
        lid = det.get("logical_track_id") or det.get("track_id")
        if lid is None:
            continue
        by_track.setdefault(lid, []).append(det)

    if not by_track:
        raise KeyframeError(
            f"No clean tracked detections found in {detections_json}. "
            "Check that detections were produced with noise filtering enabled."
        )

    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    keyframes: list[Keyframe] = []

    for lid, dets in sorted(by_track.items()):
        # sort by frame index
        dets_sorted = sorted(dets, key=lambda d: d["frame_index"])

        # three anchors — start / peak / end
        start_det = dets_sorted[0]
        peak_det  = max(dets_sorted, key=lambda d: d["confidence"])
        end_det   = dets_sorted[-1]

        # deduplicate: if start == peak or end == peak (short track), skip dupes
        selected: list[tuple[str, dict]] = [("start", start_det)]
        if peak_det["frame_index"] != start_det["frame_index"]:
            selected.append(("peak", peak_det))
        if end_det["frame_index"] != peak_det["frame_index"] and \
           end_det["frame_index"] != start_det["frame_index"]:
            selected.append(("end", end_det))

        for role, det in selected:
            source_frame = frames_dir / det["frame_file"]
            if not source_frame.exists():
                # frame might not be in the expected dir — skip gracefully
                continue

            cls = det["class_name"]
            fi  = det["frame_index"]
            kf_name = f"track_{lid:04d}_{cls}_{role}_{det['frame_file']}"
            shutil.copy(source_frame, keyframes_dir / kf_name)

            keyframes.append(Keyframe(
                logical_track_id=lid,
                class_name=cls,
                role=role,
                frame_index=fi,
                source_frame=det["frame_file"],
                keyframe_file=kf_name,
                confidence=det["confidence"],
                bbox_xyxy=det["bbox_xyxy"],
            ))

    with open(output_dir / "keyframes.json", "w", encoding="utf-8") as f:
        json.dump([asdict(k) for k in keyframes], f, indent=2)

    return keyframes


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python -m src.vision.keyframes <frames_dir> <detections.json> <output_dir>")
        sys.exit(1)

    kfs = select_keyframes(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Selected {len(kfs)} keyframes -> {sys.argv[3]}\\keyframes\\")
