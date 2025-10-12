from __future__ import annotations

from skill_core.engine import DomainState, _apply_objective_step


def _monotonic(values: list[float]) -> bool:
    return all(b <= a + 1e-6 for a, b in zip(values, values[1:]))


def test_rasch_updates_reduce_se_and_raise_theta():
    state = DomainState(level=0)
    se_values: list[float] = []
    theta_values: list[float] = []

    for _ in range(10):
        _apply_objective_step(state, 0, True, "long")
        se_values.append(state.se)
        theta_values.append(state.theta)

    assert _monotonic(se_values), "SE should decrease for repeated correct answers"
    assert all(
        b >= a - 1e-6 for a, b in zip(theta_values, theta_values[1:])
    ), "Theta should increase with consistent success"


def test_rasch_balanced_answers_keep_theta_near_zero():
    state = DomainState(level=0)
    se_values: list[float] = []

    for idx in range(12):
        correct = (idx % 2) == 0
        _apply_objective_step(state, 0, correct, "long")
        se_values.append(state.se)

    assert _monotonic(se_values), "SE should still shrink with mixed performance"
    assert abs(state.theta) < 0.3, "Alternating answers should stabilise theta around zero"

