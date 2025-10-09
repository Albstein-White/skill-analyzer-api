from __future__ import annotations

from skill_core.policy import DomainHistory, PolicyState, QuestionPolicy
from skill_core.question_bank import DOMAINS

from tests.conftest import build_synthetic_bank


def test_variant_groups_not_reused_within_session():
    bank = build_synthetic_bank()
    target_domain = DOMAINS[0]

    duplicates = [
        it
        for it in bank
        if it.domain == target_domain and it.type == "MCQ" and it.difficulty == 0
    ][:2]
    for it in duplicates:
        it.variant_group = f"{target_domain}_shared_vg"

    policy = QuestionPolicy(bank, "long")
    hist = {domain: DomainHistory() for domain in DOMAINS}
    state = PolicyState(
        run_type="long",
        theta={domain: 0.0 for domain in DOMAINS},
        se={domain: (1.5 if domain == target_domain else 0.1) for domain in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=0,
        hist=hist,
        info_history=[],
    )

    first = policy.next_item(state)
    assert first is not None and first.domain == target_domain
    vg_first = getattr(first, "variant_group", None)
    state.step += 1
    state.asked.add(first.id)
    if vg_first:
        state.seen_variant_groups.add(vg_first)
    state.hist[target_domain].obj_count += 1
    state.hist[target_domain].asked_ids.append(first.id)
    state.last_domain_id = target_domain

    second = policy.next_item(state)
    assert second is not None and second.domain == target_domain
    vg_second = getattr(second, "variant_group", None)
    assert vg_second != vg_first, "Policy should not repeat variant groups within a session"

