from __future__ import annotations

from skill_core.config import SR_START_SHIFT
import skill_core.engine as eng
from skill_core.engine import DomainState, _composite, _sr_shift_from_mean
from skill_core.question_bank import DOMAINS

from tests.conftest import build_synthetic_bank


def test_sr_weight_zero_in_composite():
    domain = DOMAINS[0]
    low_obj = DomainState(theta=-0.5, mcq_total=4, mcq_correct=1)
    sr_heavy = DomainState(theta=-0.5, mcq_total=4, mcq_correct=1, sr_sum=40.0, sr_total=40)

    score_low, _, parts_low = _composite(domain, low_obj)
    score_sr, _, parts_sr = _composite(domain, sr_heavy)

    assert abs(score_low - score_sr) < 1e-9, "SR responses must not change the composite score"
    assert parts_sr["sr"] == 100.0 and parts_low["sr"] == 0.0


def test_sr_shift_mapping_boundaries():
    assert _sr_shift_from_mean(2.0) == SR_START_SHIFT.get("low", -1)
    assert _sr_shift_from_mean(3.5) == SR_START_SHIFT.get("mid", 0)
    assert _sr_shift_from_mean(4.8) == SR_START_SHIFT.get("high", 1)


def test_start_level_defaults_and_prior_override(monkeypatch):
    bank = build_synthetic_bank(include_open=False)
    monkeypatch.setattr(eng, "load_bank", lambda: list(bank))

    monkeypatch.setattr(eng, "load_config", lambda: {})
    session_default = eng.AdaptiveSession("short")
    assert all(
        session_default.state.domains[d].level == 0 for d in DOMAINS
    ), "Default start level should be 0 when no priors"

    sr_cfg = {d: 4.8 for d in DOMAINS}
    monkeypatch.setattr(eng, "load_config", lambda: {"sr_means": sr_cfg})
    session_sr = eng.AdaptiveSession("long")
    assert all(
        session_sr.state.domains[d].level in {0, 1}
        for d in DOMAINS
    ), "SR shift must remain within ±1"

    prior_cfg = {d: 1.6 for d in DOMAINS}
    monkeypatch.setattr(
        eng,
        "load_config",
        lambda: {"sr_means": sr_cfg, "theta_priors": prior_cfg},
    )
    session_prior = eng.AdaptiveSession("long")
    assert all(
        session_prior.state.domains[d].level == 2 for d in DOMAINS
    ), "Rounded prior theta should override SR shift within level bounds"

