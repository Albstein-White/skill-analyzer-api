import pytest

from skill_core import engine as eng
from skill_core.question_bank import DOMAINS
from skill_core.types import Answer, Item

from tests.test_fail_fast import _dev_run


def _cycle_bank() -> list[Item]:
    cycle = [0, 1, 2, 3]
    items: list[Item] = []
    for domain_idx, domain in enumerate(DOMAINS):
        for offset in range(3):
            correct = cycle[(domain_idx * 3 + offset) % len(cycle)]
            items.append(
                Item(
                    id=f"{domain}_mcq_{offset}",
                    domain=domain,
                    type="MCQ",
                    text=f"{domain} MCQ #{offset}",
                    options=["A", "B", "C", "D"],
                    correct=correct,
                    difficulty=0,
                    variant_group=f"{domain}_mcq_{offset}",
                )
            )
        items.append(
            Item(
                id=f"{domain}_sr",
                domain=domain,
                type="SR",
                text=f"Rate your {domain} confidence",
                options=["0", "1", "2", "3", "4"],
                difficulty=0,
                variant_group=f"{domain}_sr",
            )
        )
        items.append(
            Item(
                id=f"{domain}_open",
                domain=domain,
                type="OPEN",
                text=f"Describe a {domain} project",
                difficulty=0,
                variant_group=f"{domain}_open",
            )
        )
    return items


@pytest.fixture
def cycle_bank(monkeypatch):
    bank = _cycle_bank()
    monkeypatch.setattr(eng, "load_bank", lambda: list(bank))
    monkeypatch.setattr(eng.random, "shuffle", lambda seq: None)
    import skill_core.policy as policy_mod

    monkeypatch.setattr(policy_mod, "DEBUG_SEED", 12345)
    return bank


def test_mcq_indexing_guard(monkeypatch, cycle_bank):
    session = eng.AdaptiveSession("short")
    mcq_items = []
    while True:
        item = session.next_item()
        if item is None:
            break
        if item.type in {"MCQ", "SJT"}:
            mcq_items.append(item)
            session.answer_current(Answer(item_id=item.id, value=3, rt_sec=30.0))
        elif item.type == "SR":
            session.answer_current(Answer(item_id=item.id, value=4, rt_sec=5.0))
        elif item.type == "OPEN":
            session.answer_current(
                Answer(
                    item_id=item.id,
                    value="I presented a detailed example to showcase my skills.",
                    rt_sec=45.0,
                )
            )
    session.finalize()

    total_mcq = sum(session.state.domains[d].mcq_total for d in DOMAINS)
    correct_mcq = sum(session.state.domains[d].mcq_correct for d in DOMAINS)
    expected_hits = sum(1 for it in mcq_items if getattr(it, "correct", None) == 3)

    assert len(mcq_items) == total_mcq == 24
    assert correct_mcq == expected_hits == 6
    assert correct_mcq / max(total_mcq, 1) == pytest.approx(expected_hits / total_mcq)

    assert all(it.options == ["A", "B", "C", "D"] for it in mcq_items)
    assert all(0 <= int(getattr(it, "correct", -1)) <= 3 for it in mcq_items)


def test_short_seed_invariants(tmp_path):
    payload_pass = _dev_run(tmp_path, {"run": "short", "mode": "pass", "seed": 11})
    assert payload_pass["steps"] == 40
    assert payload_pass.get("open_used", 0) == 0
    assert payload_pass.get("stop_reason") in (None, "")

    payload_fail = _dev_run(tmp_path, {"run": "short", "mode": "fail", "seed": 12})
    assert payload_fail["steps"] < 40
    assert payload_fail.get("stop_reason") == "fail_fast_short"
    assert payload_fail.get("open_used", 0) == 0
