from __future__ import annotations

from skill_core.config import OBJ_MAX_LONG
from skill_core.engine import DomainState, _apply_objective_step


def test_long_run_reaches_top_level_with_three_of_four_rule():
    state = DomainState(level=-1)
    sequence = [
        (-1, True),
        (-1, True),
        (-1, True),
        (-1, False),
        (0, True),
        (0, True),
        (0, True),
        (0, False),
        (1, True),
        (1, True),
        (1, True),
        (1, False),
        (2, True),
        (2, True),
        (2, True),
        (2, True),
    ]

    for idx, (difficulty, correct) in enumerate(sequence, start=1):
        metrics = _apply_objective_step(state, difficulty, correct, "long")
        if idx < 4:
            assert metrics["level_after"] == -1, "Long run should require 3/4 window before promoting"
        if idx == 4:
            assert metrics["level_before"] == -1 and metrics["level_after"] == 0

    assert state.level == 2, "Long ladder should climb to +2"
    assert state.b_stable == 2, "Sustained success at the top should mark level +2 as stable"
    assert state.obj_count <= OBJ_MAX_LONG, "Objective count must respect long-run cap"

