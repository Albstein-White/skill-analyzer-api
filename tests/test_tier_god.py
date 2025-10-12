from types import SimpleNamespace

import skill_core.config as cfg
import skill_core.engine as engine


def _state(**overrides):
    base = {
        "b_stable": 2,
        "se": 0.18,
        "items_by_level": {2: 5},
        "accuracy_by_level": {2: 0.85},
        "open_count": 2,
        "open_ratings": [0.86, 0.80],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_god_gate_passes_with_all_requirements():
    state = _state()
    tier = engine.map_tier(9.9, "long", state)
    assert tier == "GOD"
    gate = getattr(state, "_tier_meta", {}).get("god_gate", {})
    assert gate.get("status") == "pass"


def test_god_gate_requires_low_se():
    state = _state(se=0.26)
    tier = engine.map_tier(9.9, "long", state)
    assert tier == "SS"
    gate = getattr(state, "_tier_meta", {}).get("god_gate", {})
    assert gate.get("status") == "near"
    assert gate.get("se_ok") is False
    assert "se_high" in gate.get("reasons", [])


def test_god_gate_requires_two_open():
    state = _state(open_count=1, open_ratings=[0.86])
    tier = engine.map_tier(9.9, "long", state)
    assert tier == "SS"
    gate = getattr(state, "_tier_meta", {}).get("god_gate", {})
    assert gate.get("open_count_ok") is False
    assert "open_count_low" in gate.get("reasons", [])
    assert gate.get("status") == "near"


def test_god_gate_honours_staging_overrides(monkeypatch):
    monkeypatch.setattr(cfg, "STAGING_PROFILE", True, raising=False)
    monkeypatch.setattr(cfg, "TEST_MODE", True, raising=False)
    monkeypatch.setattr(cfg, "TEST_GOD_MAX_SE", 0.60, raising=False)
    monkeypatch.setattr(cfg, "TEST_GOD_MIN_NORM", 9.4, raising=False)

    state = _state(se=0.55)
    tier = engine.map_tier(9.45, "long", state)
    assert tier == "GOD"
    gate = getattr(state, "_tier_meta", {}).get("god_gate", {})
    assert gate.get("status") == "pass"
    assert "se_high" not in gate.get("reasons", [])
