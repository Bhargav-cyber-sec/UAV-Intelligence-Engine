"""
Central config loader. Every later phase reads its settings through here
instead of hardcoding thresholds/paths, so your guide can tune behavior
by editing config/*.yaml without touching code.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class Config:
    ontology: dict
    paths: dict

    @property
    def mission_phases(self) -> list[str]:
        return self.ontology["mission_phases"]

    @property
    def event_types(self) -> list[dict]:
        return self.ontology["event_types"]

    @property
    def scoring_weights(self) -> dict:
        return self.ontology["scoring"]["weights"]

    @property
    def candidate_threshold(self) -> float:
        return self.ontology["scoring"]["candidate_threshold"]

    @property
    def detection_fps(self) -> float:
        return self.ontology["sampling"]["detection_fps"]

    @property
    def yolo_inference_size(self) -> int:
        return self.ontology["sampling"]["yolo_inference_size"]

    @property
    def ffmpeg_bin(self) -> Path:
        local_bin = REPO_ROOT / self.paths["tools"]["ffmpeg_bin"]
        if (local_bin / "ffmpeg.exe").exists():
            return local_bin
        parent_bin = REPO_ROOT.parent / "Phase1_Environment_Setup" / self.paths["tools"]["ffmpeg_bin"]
        if (parent_bin / "ffmpeg.exe").exists():
            return parent_bin
        return local_bin

    @property
    def ffmpeg_exe(self) -> Path:
        local_exe = self.ffmpeg_bin / "ffmpeg.exe"
        if local_exe.exists():
            return local_exe
        import shutil
        which = shutil.which("ffmpeg")
        return Path(which) if which else local_exe

    @property
    def ffprobe_exe(self) -> Path:
        local_exe = self.ffmpeg_bin / "ffprobe.exe"
        if local_exe.exists():
            return local_exe
        import shutil
        which = shutil.which("ffprobe")
        return Path(which) if which else local_exe



def _load_yaml(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> Config:
    ontology = _load_yaml("ontology.yaml")
    paths = _load_yaml("paths.yaml")
    return Config(ontology=ontology, paths=paths)


if __name__ == "__main__":
    cfg = load_config()
    print("Mission phases:", cfg.mission_phases)
    print("Event types:", [e["id"] for e in cfg.event_types])
    print("Scoring weights:", cfg.scoring_weights)
    print("Detection sampling:", cfg.detection_fps, "fps @", cfg.yolo_inference_size, "px")