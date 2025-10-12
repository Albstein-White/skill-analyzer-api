from __future__ import annotations

import json

from skill_core import engine as eng
from skill_core.config import CAP_LONG, OBJ_MIN_LONG
from skill_core.policy import DomainHistory, PolicyState, QuestionPolicy
from skill_core.question_bank import DOMAINS
from skill_core.report_html import export_report_html

from tests.conftest import build_synthetic_bank


def test_objective_minima_are_fulfilled_before_sr_or_open():
    bank = build_synthetic_bank()
    policy = QuestionPolicy(bank, "long")
    hist = {domain: DomainHistory() for domain in DOMAINS}
    state = PolicyState(
        run_type="long",
        theta={domain: 0.0 for domain in DOMAINS},
        se={domain: 1.0 for domain in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=0,
        hist=hist,
        info_history=[],
    )

    for _ in range(OBJ_MIN_LONG * len(DOMAINS)):
        item = policy.next_item(state)
        assert item is not None
        assert item.type in {"MCQ", "SJT"}
        state.step += 1
        state.asked.add(item.id)
        vg = getattr(item, "variant_group", None)
        if vg:
            state.seen_variant_groups.add(vg)
        domain_hist = state.hist[item.domain]
        domain_hist.obj_count += 1
        domain_hist.asked_ids.append(item.id)
        state.last_domain_id = item.domain

    assert all(hist[d].obj_count >= OBJ_MIN_LONG for d in DOMAINS)


def test_finalize_marks_incomplete_when_cap_hits_before_minima(tmp_path, monkeypatch):
    bank = build_synthetic_bank()
    monkeypatch.setattr(eng, "load_bank", lambda: list(bank))

    session = eng.AdaptiveSession("long")
    for domain in DOMAINS:
        session.state.domains[domain].obj_count = OBJ_MIN_LONG - 1
    session._step = CAP_LONG

    result = session.finalize()
    meta = getattr(result, "meta", {})
    assert meta.get("incomplete") is True
    assert meta.get("incomplete_reason") == "CAP_BEFORE_MINIMA"
    shortfalls = meta.get("shortfalls")
    assert shortfalls and all(val == 1 for val in shortfalls.values())

    payload = {
        "run_type": result.run_type,
        "summary": result.summary,
        "meta": meta,
        "domain_scores": [json.loads(json.dumps(ds.__dict__)) for ds in result.domain_scores],
    }

    out_path = tmp_path / "report.html"
    export_report_html(payload, str(out_path))
    html = out_path.read_text(encoding="utf-8")

    assert "Run hit step cap before completing minimum objectives" in html
    for domain in DOMAINS:
        assert f"{domain}: needs 1 more objective item" in html

