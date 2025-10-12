from types import SimpleNamespace

from skill_core.config import SHORT_TIER_CAP
from skill_core.engine import map_tier


def _state(**overrides):
    base = {
        "b_stable": 2,
        "se": 0.18,
        "items_by_level": {2: 6},
        "accuracy_by_level": {2: 0.9},
        "open_count": 2,
        "open_ratings": [0.9, 0.85],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_short_cap_blocks_ss():
    state = _state()
    tier = map_tier(9.7, "short", state)
    assert tier == SHORT_TIER_CAP
    meta = getattr(state, "_tier_meta", {})
    assert meta.get("short_cap") is True
    assert meta.get("base_tier") in {"S", "SS"}


def test_short_cap_blocks_perfect_session():
    state = _state()
    tier = map_tier(10.0, "short", state)
    assert tier == SHORT_TIER_CAP
    meta = getattr(state, "_tier_meta", {})
    assert meta.get("short_cap") is True
