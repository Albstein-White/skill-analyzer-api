from __future__ import annotations

from skill_core.config import OPEN_INFO_CAP_RATIO
from skill_core.engine import DomainState, _apply_open_step


def _prep(theta: float, obj_info: float) -> DomainState:
    state = DomainState()
    state.theta = theta
    state.obj_info_total = obj_info
    state.info_total = obj_info + 1.0
    return state


def test_open_residual_updates_and_caps():
    positive = _prep(0.8, 10.0)
    metrics_high = _apply_open_step(
        positive,
        difficulty=1,
        score=0.70,
        word_count=180,
        latency_ms=2500,
        rubric_conf=0.8,
    )

    assert metrics_high["dtheta"] > 0.0, "Residual above expectation should raise theta"

    smaller = _prep(0.8, 10.0)
    metrics_low = _apply_open_step(
        smaller,
        difficulty=-1,
        score=0.70,
        word_count=180,
        latency_ms=2500,
        rubric_conf=0.8,
    )

    assert metrics_low["dtheta"] < metrics_high["dtheta"]
    cap = OPEN_INFO_CAP_RATIO * smaller.obj_info_total
    assert smaller.open_info_total <= cap + 1e-9

    ignored = _prep(0.8, 10.0)
    before_theta = ignored.theta
    metrics_ignored = _apply_open_step(
        ignored,
        difficulty=1,
        score=0.9,
        word_count=50,
        latency_ms=200,
        rubric_conf=0.9,
    )
    assert metrics_ignored.get("ignored") == "latency"
    assert abs(ignored.theta - before_theta) < 1e-9

