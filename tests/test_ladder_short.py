from __future__ import annotations

from skill_core.engine import DomainState, _apply_objective_step


def test_short_run_stays_within_band_and_promotes_to_plus_one():
    state = DomainState(level=-1)
    sequence = [
        (-1, True),
        (-1, True),
        (-1, False),  # two of three correct -> promote to 0
        (0, True),
        (0, True),
        (0, False),   # promotion to +1 after three items at level 0
        (1, True),
        (1, False),
        (1, True),
    ]

    seen_levels = []
    for idx, (difficulty, correct) in enumerate(sequence, start=1):
        metrics = _apply_objective_step(state, difficulty, correct, "short")
        seen_levels.append(metrics["level_after"])
        assert -1 <= metrics["level_after"] <= 1, "Short ladder must remain within -1..+1 band"
        if idx == 3:
            assert metrics["level_before"] == -1 and metrics["level_after"] == 0
        if idx == 6:
            assert metrics["level_before"] == 0 and metrics["level_after"] == 1

    assert state.level == 1, "Short ladder should peak at +1 under new policy"
    assert state.b_stable <= 1, "Stable level cannot exceed band"
    assert any(level == 1 for level in seen_levels), "Progression should capture promotion to +1"

