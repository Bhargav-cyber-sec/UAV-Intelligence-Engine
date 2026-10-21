"""
Tests for src.media.probe and src.media.sample.

These generate their own tiny synthetic clips via the bundled ffmpeg
(testsrc pattern, 2 seconds) rather than requiring real UAV footage -
per docs/known_limitations.md, real footage isn't available yet, and
this module's job (report duration/resolution/audio, extract frames)
doesn't depend on what's actually in the frame.
"""
from __future__ import annotations

import subprocess

import pytest

from src.config import load_config
from src.media.probe import probe, ProbeError
from src.media.sample import sample_frames, SampleError


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def silent_clip(tmp_path_factory, cfg):
    """2s, 320x240, no audio track."""
    out = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        [
            str(cfg.ffmpeg_exe), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            str(out),
        ],
        capture_output=True, check=True,
    )
    return out


@pytest.fixture(scope="module")
def clip_with_audio(tmp_path_factory, cfg):
    """2s, 320x240, with a silent stereo audio track."""
    out = tmp_path_factory.mktemp("media") / "with_audio.mp4"
    subprocess.run(
        [
            str(cfg.ffmpeg_exe), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-shortest",
            str(out),
        ],
        capture_output=True, check=True,
    )
    return out


def test_probe_reports_no_audio(silent_clip):
    info = probe(silent_clip)
    assert info.has_audio is False
    assert info.audio_codec is None
    assert info.width == 320
    assert info.height == 240


def test_probe_reports_audio_present(clip_with_audio):
    info = probe(clip_with_audio)
    assert info.has_audio is True
    assert info.audio_codec is not None
    assert info.audio_sample_rate == 44100


def test_probe_reports_duration_close_to_source(silent_clip):
    info = probe(silent_clip)
    assert 1.8 <= info.duration <= 2.2


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(ProbeError):
        probe(tmp_path / "does_not_exist.mp4")


def test_sample_frames_extracts_expected_count(silent_clip, tmp_path, cfg):
    out_dir = tmp_path / "frames"
    frames = sample_frames(silent_clip, out_dir)
    expected = round(cfg.detection_fps * 2)  # 2s clip at detection_fps
    assert abs(len(frames) - expected) <= 1
    assert all(f.exists() for f in frames)


def test_sample_frames_missing_file_raises(tmp_path):
    with pytest.raises(SampleError):
        sample_frames(tmp_path / "nope.mp4", tmp_path / "out")


def test_mov_clip_probing_and_sampling(tmp_path_factory, tmp_path, cfg):
    """Verify probe and frame sampling work seamlessly on QuickTime .mov files."""
    mov_clip = tmp_path_factory.mktemp("media_mov") / "test_drone.mov"
    subprocess.run(
        [
            str(cfg.ffmpeg_exe), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=640x480:rate=10",
            str(mov_clip),
        ],
        capture_output=True, check=True,
    )

    info = probe(mov_clip)
    assert info.filename == "test_drone.mov"
    assert info.width == 640
    assert info.height == 480
    assert 1.8 <= info.duration <= 2.2

    out_dir = tmp_path / "mov_frames"
    frames = sample_frames(mov_clip, out_dir)
    assert len(frames) > 0