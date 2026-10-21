"""
Tests for src.vision.keyframes.

Builds a synthetic detections.json directly (rather than running full
YOLO inference again) with two tracks across three frames, where the
highest-confidence frame for each track is deliberately not frame 0 -
this is what actually exercises the "pick the best, not just the
first" logic, which repeating the same frame five times (as
test_vision.py does for the tracking tests) would not catch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.vision.keyframes import select_keyframes, KeyframeError


@pytest.fixture
def frames_and_detections(tmp_path):
    """3 blank frames + a hand-built detections.json with 2 tracks,
    where each track's best-confidence detection is on a different frame."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    from PIL import Image
    for i in range(1, 4):
        Image.new("RGB", (32, 32)).save(frames_dir / f"frame_{i:06d}.jpg")

    detections = [
        {"frame_index": 0, "frame_file": "frame_000001.jpg", "track_id": 1,
         "class_id": 0, "class_name": "person", "confidence": 0.55, "bbox_xyxy": [0, 0, 1, 1]},
        {"frame_index": 1, "frame_file": "frame_000002.jpg", "track_id": 1,
         "class_id": 0, "class_name": "person", "confidence": 0.91, "bbox_xyxy": [0, 0, 1, 1]},
        {"frame_index": 2, "frame_file": "frame_000003.jpg", "track_id": 1,
         "class_id": 0, "class_name": "person", "confidence": 0.70, "bbox_xyxy": [0, 0, 1, 1]},
        {"frame_index": 0, "frame_file": "frame_000001.jpg", "track_id": 2,
         "class_id": 2, "class_name": "car", "confidence": 0.80, "bbox_xyxy": [0, 0, 1, 1]},
        {"frame_index": 1, "frame_file": "frame_000002.jpg", "track_id": 2,
         "class_id": 2, "class_name": "car", "confidence": 0.60, "bbox_xyxy": [0, 0, 1, 1]},
    ]
    det_path = tmp_path / "detections.json"
    with open(det_path, "w") as f:
        json.dump(detections, f)

    return frames_dir, det_path


def test_picks_highest_confidence_frame_per_track(frames_and_detections, tmp_path):
    frames_dir, det_path = frames_and_detections
    keyframes = select_keyframes(frames_dir, det_path, tmp_path / "out")
    peak_by_track = {k.track_id: k for k in keyframes if k.role == "peak"}

    assert peak_by_track[1].frame_index == 1       # 0.91 was the best for track 1
    assert peak_by_track[1].confidence == 0.91
    # For track 2, frame 0 has 0.80 (which is both start and peak)
    track2_best = max([k for k in keyframes if k.track_id == 2], key=lambda x: x.confidence)
    assert track2_best.frame_index == 0            # 0.80 was the best for track 2
    assert track2_best.confidence == 0.80


def test_keyframes_per_track_count(frames_and_detections, tmp_path):
    frames_dir, det_path = frames_and_detections
    keyframes = select_keyframes(frames_dir, det_path, tmp_path / "out")
    track_ids = {k.track_id for k in keyframes}
    assert track_ids == {1, 2}  # exactly 2 distinct tracks represented
    for tid in track_ids:
        track_kfs = [k for k in keyframes if k.track_id == tid]
        assert 1 <= len(track_kfs) <= 3  # up to 3 keyframes per track (start/peak/end)



def test_keyframe_files_actually_written(frames_and_detections, tmp_path):
    frames_dir, det_path = frames_and_detections
    out_dir = tmp_path / "out"
    keyframes = select_keyframes(frames_dir, det_path, out_dir)
    for kf in keyframes:
        assert (out_dir / "keyframes" / kf.keyframe_file).exists()
    assert (out_dir / "keyframes.json").exists()


def test_missing_detections_json_raises(tmp_path):
    with pytest.raises(KeyframeError):
        select_keyframes(tmp_path, tmp_path / "no_such_file.json", tmp_path / "out")


def test_detections_with_no_tracks_raises(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    det_path = tmp_path / "detections.json"
    with open(det_path, "w") as f:
        json.dump([{"frame_index": 0, "frame_file": "x.jpg", "track_id": None,
                     "class_id": 0, "class_name": "person", "confidence": 0.5,
                     "bbox_xyxy": [0, 0, 1, 1]}], f)
    with pytest.raises(KeyframeError):
        select_keyframes(frames_dir, det_path, tmp_path / "out")