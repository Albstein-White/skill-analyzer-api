from __future__ import annotations

from skill_core.engine import DomainState, _apply_objective_step


def test_first_miss_does_not_demote_and_subsequent_misses_do():
    state = DomainState(level=0)

    for difficulty, correct in [
        (0, True),
        (0, True),
        (0, False),
    ]:
        _apply_objective_step(state, difficulty, correct, "short")

    assert state.level == 1, "Initial streak should promote to level +1"

    _apply_objective_step(state, 1, False, "short")
    assert state.level == 1, "First item at a new level cannot demote"

    before_drop = state.level
    _apply_objective_step(state, 1, False, "short")
    metrics = _apply_objective_step(state, 1, False, "short")
    assert state.level == 0, "Two misses in the last three should demote by one level"
    assert metrics["level_before"] == before_drop and metrics["level_after"] == before_drop - 1

