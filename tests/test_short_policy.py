import pytest

from skill_core import engine as eng
from skill_core.policy import DomainHistory, PolicyState, QuestionPolicy
from skill_core.config import (
    CAP_SHORT,
    SHORT_OBJ_MIN,
    SHORT_OBJ_MAX,
    SHORT_EXTRA_TOTAL,
    SHORT_SR_PER_DOMAIN,
    SHORT_LEVELS,
    SHORT_START_LEVEL,
    TIER_NAMES,
)
from skill_core.question_bank import DOMAINS
from skill_core.types import Answer

from tests.conftest import build_synthetic_bank


def _run_short_session(monkeypatch):
    bank = build_synthetic_bank(include_open=False)
    monkeypatch.setattr(eng, "load_bank", lambda: list(bank))
    monkeypatch.setattr(eng.random, "shuffle", lambda seq: None)

    import skill_core.policy as policy_mod

    monkeypatch.setattr(policy_mod, "DEBUG_SEED", 12345)

    session = eng.AdaptiveSession("short")
    initial_levels = {dom: session.state.domains[dom].level for dom in DOMAINS}
    served = []

    while True:
        item = session.next_item()
        if item is None:
            break
        served.append(item)
        if item.type in {"MCQ", "SJT"}:
            value = int(getattr(item, "correct", 0) or 0)
            answer = Answer(item_id=item.id, value=value, rt_sec=30.0)
        elif item.type == "SR":
            answer = Answer(item_id=item.id, value=4, rt_sec=5.0)
        else:
            answer = Answer(item_id=item.id, value="", rt_sec=45.0)
        session.answer_current(answer)

    result = session.finalize()
    return session, served, result, initial_levels


@pytest.fixture
def short_session(monkeypatch):
    return _run_short_session(monkeypatch)


def test_minima_counts(short_session):
    session, served, _, _ = short_session
    for domain in DOMAINS:
        st = session.state.domains[domain]
        assert st.obj_count >= SHORT_OBJ_MIN
        assert st.obj_count <= SHORT_OBJ_MAX
    obj_total = sum(session.state.domains[d].obj_count for d in DOMAINS)
    assert obj_total >= SHORT_OBJ_MIN * len(DOMAINS)


def test_sr_counts(short_session):
    session, served, _, _ = short_session
    obj_progress = {dom: 0 for dom in DOMAINS}
    for item in served:
        if item.type in {"MCQ", "SJT"}:
            obj_progress[item.domain] += 1
        elif item.type == "SR":
            assert (
                obj_progress[item.domain] >= SHORT_OBJ_MIN
            ), "SR must follow objective minima"
    for domain in DOMAINS:
        assert session.state.domains[domain].sr_count == SHORT_SR_PER_DOMAIN


def test_level_banding(short_session):
    session, served, _, initial_levels = short_session
    for domain, level in initial_levels.items():
        assert level == SHORT_START_LEVEL, "Short runs should start at configured level"
    allowed = set(SHORT_LEVELS)
    for item in served:
        if item.type in {"MCQ", "SJT"}:
            assert int(getattr(item, "difficulty", 0)) in allowed


def test_extra_allocation(monkeypatch):
    bank = build_synthetic_bank(include_open=False)
    policy = QuestionPolicy(bank, "short")
    hist = {dom: DomainHistory() for dom in DOMAINS}
    se_values = {
        dom: 1.0 - idx * 0.05 for idx, dom in enumerate(DOMAINS)
    }
    step_base = (SHORT_OBJ_MIN + SHORT_SR_PER_DOMAIN) * len(DOMAINS)
    for dom in DOMAINS:
        hist[dom].obj_count = SHORT_OBJ_MIN
        hist[dom].sr_count = SHORT_SR_PER_DOMAIN
        hist[dom].level = 0
    state = PolicyState(
        run_type="short",
        theta={dom: 0.0 for dom in DOMAINS},
        se=se_values,
        asked=set(),
        seen_variant_groups=set(),
        step=step_base,
        hist=hist,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type="MCQ",
    )

    extras_taken = {dom: 0 for dom in DOMAINS}
    expected_total = min(
        SHORT_EXTRA_TOTAL,
        max(0, CAP_SHORT - step_base),
    )

    while sum(extras_taken.values()) < expected_total:
        item = policy.next_item(state)
        assert item is not None, "Policy should allocate expected extras"
        assert item.type in {"MCQ", "SJT"}
        dom_hist = state.hist[item.domain]
        dom_hist.obj_count += 1
        extras_taken[item.domain] = max(0, dom_hist.obj_count - SHORT_OBJ_MIN)
        dom_hist.asked_ids.append(item.id)
        state.asked.add(item.id)
        vg = getattr(item, "variant_group", None)
        if vg:
            state.seen_variant_groups.add(vg)
        state.step += 1
        state.last_domain_id = item.domain
        state.last_item_type = item.type

    ordered = sorted(DOMAINS, key=lambda d: se_values[d], reverse=True)
    top_four = ordered[:4]
    for dom in top_four:
        assert extras_taken[dom] <= min(2, SHORT_OBJ_MAX - SHORT_OBJ_MIN)
    others = [dom for dom in DOMAINS if dom not in top_four]
    for dom in others:
        assert extras_taken[dom] == 0
    assert sum(extras_taken.values()) == expected_total
    for dom in DOMAINS:
        assert state.hist[dom].obj_count <= SHORT_OBJ_MAX


def test_short_extras_top4_only(short_session):
    session, _, result, _ = short_session
    extras = []
    for domain in DOMAINS:
        st = session.state.domains[domain]
        assert SHORT_OBJ_MIN <= st.obj_count <= SHORT_OBJ_MAX
        extras.append(max(0, st.obj_count - SHORT_OBJ_MIN))

    extras_total = sum(extras)
    expected_total = min(
        SHORT_EXTRA_TOTAL,
        max(0, CAP_SHORT - (SHORT_OBJ_MIN + SHORT_SR_PER_DOMAIN) * len(DOMAINS)),
    )
    assert extras_total == expected_total

    domains_with_extras = sum(1 for val in extras if val > 0)
    assert domains_with_extras <= 4
    assert all(val <= (SHORT_OBJ_MAX - SHORT_OBJ_MIN) for val in extras)


def test_short_total_steps_approx40(short_session):
    session, _, result, _ = short_session
    expected_extras = min(
        SHORT_EXTRA_TOTAL,
        max(0, CAP_SHORT - (SHORT_OBJ_MIN + SHORT_SR_PER_DOMAIN) * len(DOMAINS)),
    )
    expected_steps = (
        SHORT_OBJ_MIN * len(DOMAINS)
        + SHORT_SR_PER_DOMAIN * len(DOMAINS)
        + expected_extras
    )
    assert session._step == expected_steps
    assert result.summary.get("steps_total", 0) == expected_steps


def test_cap(short_session):
    session, _, result, _ = short_session
    assert session._step <= CAP_SHORT
    assert result.summary.get("steps_total", 0) <= CAP_SHORT


def test_tier_cap(short_session):
    _, _, result, _ = short_session
    cap_index = TIER_NAMES.index("A")
    for score in result.domain_scores:
        tier_index = TIER_NAMES.index(score.tier)
        assert tier_index <= cap_index
