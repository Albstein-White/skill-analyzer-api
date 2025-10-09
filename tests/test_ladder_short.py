from __future__ import annotations

from skill_core.config import OBJ_MAX_SHORT
from skill_core.engine import DomainState, _apply_objective_step


def test_short_run_reaches_top_level_with_pass_rule():
    state = DomainState(level=-1)
    sequence = [
        (-1, True),
        (-1, True),
        (-1, False),
        (0, True),
        (0, True),
        (0, False),
        (1, True),
        (1, True),
        (1, False),
        (2, True),
        (2, True),
        (2, True),
    ]

    history = []
    for idx, (difficulty, correct) in enumerate(sequence, start=1):
        metrics = _apply_objective_step(state, difficulty, correct, "short")
        history.append(metrics)
        if idx < 3:
            assert metrics["level_after"] == -1, "Short run should require 2/3 window before promoting"
        if idx == 3:
            assert metrics["level_before"] == -1 and metrics["level_after"] == 0

    assert state.level == 2, "Ladder should climb to level +2"
    assert state.b_stable == 2, "Stable level should reflect sustained success at +2"
    assert state.obj_count <= OBJ_MAX_SHORT, "Objective count must stay within short cap"
    assert any(entry["level_after"] == 2 for entry in history), "History should capture promotion to +2"

