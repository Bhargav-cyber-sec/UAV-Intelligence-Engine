"""
Tests for src.vision.detect and src.vision.keyframes (Week 2 reliability rewrite).

Uses ultralytics' own bus.jpg test asset (repeated frames) rather than real UAV
footage — the contract under test is track continuity, noise filtering, ghost
reconciliation, and correct JSON output, not what's actually in the frame.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import ultralytics

from src.config import load_config
from src.vision.detect import detect_and_track, DetectionError
from src.vision.keyframes import select_keyframes, KeyframeError

ASSET_IMAGE = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def repeated_frames(tmp_path_factory):
    """5 identical frames of ultralytics' own bus.jpg test asset."""
    frames_dir = tmp_path_factory.mktemp("vision_frames")
    for i in range(1, 6):
        shutil.copy(ASSET_IMAGE, frames_dir / f"frame_{i:06d}.jpg")
    return frames_dir


# ─── detect_and_track tests ──────────────────────────────────────────────────

def test_detect_finds_expected_classes(repeated_frames, tmp_path, cfg):
    detections, tracks = detect_and_track(repeated_frames, tmp_path / "out", cfg=cfg)
    class_names = {d.class_name for d in detections}
    # Compatible with both COCO ('person', 'bus') and VisDrone ('pedestrian', 'car', 'bus')
    assert any(c in class_names for c in ("person", "pedestrian", "bus", "car"))


def test_tracks_have_enriched_fields(repeated_frames, tmp_path, cfg):
    """Track dataclass must carry the new richer fields added in reliability rewrite."""
    _, tracks = detect_and_track(repeated_frames, tmp_path / "out_rich", cfg=cfg)
    assert len(tracks) >= 1
    for t in tracks:
        assert hasattr(t, "logical_track_id")
        assert hasattr(t, "duration_s")
        assert hasattr(t, "max_confidence")
        assert hasattr(t, "first_bbox")
        assert hasattr(t, "last_bbox")
        assert hasattr(t, "area_px_mean")
        assert t.mean_confidence <= t.max_confidence + 1e-6
        assert t.duration_s >= 0.0
        assert len(t.first_bbox) == 4
        assert len(t.last_bbox) == 4


def test_noise_tracks_excluded_from_tracks_json(repeated_frames, tmp_path, cfg):
    """tracks.json must contain only tracks that met the min_track_frames threshold."""
    _, tracks = detect_and_track(repeated_frames, tmp_path / "out_noise", cfg=cfg)
    min_frames = cfg.ontology.get("detection", {}).get("min_track_frames", 3)
    for t in tracks:
        assert t.frame_count >= min_frames, (
            f"Track {t.logical_track_id} has {t.frame_count} frames "
            f"but min_track_frames={min_frames}"
        )


def test_detection_summary_written(repeated_frames, tmp_path, cfg):
    out_dir = tmp_path / "out_summary"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    summary_path = out_dir / "detection_summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert "total_tracks" in summary
    assert "noise_rate_pct" in summary
    assert "frames_with_clean_detections" in summary


def test_output_files_written(repeated_frames, tmp_path, cfg):
    out_dir = tmp_path / "out_files"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    assert (out_dir / "detections.json").exists()
    assert (out_dir / "tracks.json").exists()
    assert (out_dir / "detection_summary.json").exists()


def test_detections_json_has_logical_track_id(repeated_frames, tmp_path, cfg):
    out_dir = tmp_path / "out_lid"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    with open(out_dir / "detections.json") as f:
        dets = json.load(f)
    assert len(dets) > 0
    for d in dets:
        assert "logical_track_id" in d
        assert "noise" in d


def test_missing_frames_dir_raises(tmp_path, cfg):
    with pytest.raises(DetectionError):
        detect_and_track(tmp_path / "no_such_dir", tmp_path / "out", cfg=cfg)


def test_missing_weights_raises(repeated_frames, tmp_path, cfg, monkeypatch):
    monkeypatch.setitem(cfg.paths["models"], "yolo_weights", "models/yolo/does_not_exist.pt")
    with pytest.raises(DetectionError):
        detect_and_track(repeated_frames, tmp_path / "out_bad_weights", cfg=cfg)


def test_sahi_sliced_detection(repeated_frames, tmp_path, cfg):
    """SAHI mode must produce valid JSON and persistent track IDs (not one-per-tile)."""
    cfg_copy = load_config()
    cfg_copy.ontology["detection"]["sahi"] = {
        "enabled": True,
        "slice_height": 320,
        "slice_width": 320,
        "overlap_ratio": 0.1,
        "nms_threshold": 0.45,
    }
    out_dir = tmp_path / "sahi_out"
    detections, tracks = detect_and_track(repeated_frames, out_dir, cfg=cfg_copy)
    assert (out_dir / "detections.json").exists()
    assert (out_dir / "tracks.json").exists()
    assert len(detections) >= 1
    # In SAHI mode tracks should still have persistent IDs across frames —
    # the number of tracks must be << total detections (not one track per det)
    assert len(tracks) < len(detections), (
        "SAHI produced one track per detection — IOU tracker not working"
    )


# ─── keyframes tests ──────────────────────────────────────────────────────────

def test_keyframes_three_per_track(repeated_frames, tmp_path, cfg):
    """select_keyframes must extract up to 3 keyframes per clean logical track."""
    out_dir = tmp_path / "kf_out"
    _, tracks = detect_and_track(repeated_frames, out_dir, cfg=cfg)
    if not tracks:
        pytest.skip("no clean tracks — nothing to keyframe")
    kfs = select_keyframes(repeated_frames, out_dir / "detections.json", out_dir)
    # each track has at most 3 keyframes and at least 1
    from collections import Counter
    counts = Counter(k.logical_track_id for k in kfs)
    for tid, count in counts.items():
        assert 1 <= count <= 3, f"Track {tid} has {count} keyframes (expected 1–3)"


def test_keyframes_roles_are_valid(repeated_frames, tmp_path, cfg):
    """Every keyframe must have a role of start, peak, or end."""
    out_dir = tmp_path / "kf_roles"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    kfs = select_keyframes(repeated_frames, out_dir / "detections.json", out_dir)
    valid_roles = {"start", "peak", "end"}
    for k in kfs:
        assert k.role in valid_roles, f"Unexpected keyframe role: {k.role}"


def test_keyframes_files_exist(repeated_frames, tmp_path, cfg):
    """Every keyframe referenced in keyframes.json must be physically on disk."""
    out_dir = tmp_path / "kf_files"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    kfs = select_keyframes(repeated_frames, out_dir / "detections.json", out_dir)
    for k in kfs:
        assert (out_dir / "keyframes" / k.keyframe_file).exists(), (
            f"Keyframe file missing: {k.keyframe_file}"
        )


def test_keyframes_json_written(repeated_frames, tmp_path, cfg):
    out_dir = tmp_path / "kf_json"
    detect_and_track(repeated_frames, out_dir, cfg=cfg)
    select_keyframes(repeated_frames, out_dir / "detections.json", out_dir)
    assert (out_dir / "keyframes.json").exists()
    with open(out_dir / "keyframes.json") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 1
