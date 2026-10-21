from src.config import load_config


def test_load_config_has_mission_phases():
    cfg = load_config()
    assert "takeoff" in cfg.mission_phases
    assert "landing" in cfg.mission_phases


def test_event_types_present():
    cfg = load_config()
    ids = [e["id"] for e in cfg.event_types]
    assert "object_appearance" in ids
    assert "speech_event" in ids


def test_scoring_weights_sum_close_to_one():
    cfg = load_config()
    total = sum(cfg.scoring_weights.values())
    assert 0.95 <= total <= 1.05
