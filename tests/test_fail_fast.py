import importlib
import os
import sys

from fastapi.testclient import TestClient

from skill_core.config import (
    CAP_LONG,
    LONG_FAIL_FAST_WINDOW,
    LONG_EARLY_STOP_AFTER,
    LONG_FAIL_FAST_MAX_CORRECT,
    LONG_MIN_INFO_GAIN,
    OBJ_MIN_LONG,
    SHORT_OBJ_MIN,
    SHORT_FAIL_FAST_WINDOW,
    SHORT_FAIL_FAST_MAX_CORRECT,
    SHORT_SR_PER_DOMAIN,
    SR_PER_DOMAIN_LONG,
)
from skill_core.engine import AdaptiveSession
from skill_core.policy import DomainHistory, PolicyState, QuestionPolicy
from skill_core.question_bank import DOMAINS

_MODULES = [
    "skill_core.config",
    "skill_core.plan",
    "skill_core.engine",
    "skill_core.policy",
    "api.storage",
    "api.app",
]


def _reload_app(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ.setdefault("DEBUG_SEED", "12345")
    for name in _MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            __import__(name)
    storage = sys.modules["api.storage"]
    app_module = sys.modules["api.app"]
    return storage, app_module


def _dev_run(tmp_path, params):
    os.environ["STAGING_PROFILE"] = "1"
    try:
        _, app_module = _reload_app(tmp_path)
        with TestClient(app_module.app) as client:
            response = client.get("/dev/run", params=params)
        assert response.status_code == 200
        return response.json()
    finally:
        os.environ.pop("STAGING_PROFILE", None)
        _reload_app(tmp_path)


def test_short_always_wrong_stops_early(tmp_path):
    payload = _dev_run(tmp_path, {"run": "short", "mode": "fail", "seed": 11})
    expected_steps = (SHORT_OBJ_MIN + SHORT_SR_PER_DOMAIN) * len(DOMAINS)
    assert expected_steps <= payload["steps"] < 40
    assert payload.get("open_used", 0) == 0
    assert payload.get("stop_reason") == "fail_fast_short"
    assert payload.get("ff_win_obj_len_short") == SHORT_FAIL_FAST_WINDOW


def test_short_pass_unchanged(tmp_path):
    payload = _dev_run(tmp_path, {"run": "short", "mode": "pass", "seed": 11})
    assert payload["steps"] == 40
    assert payload.get("cap", "").startswith("40/")
    assert payload.get("stop_reason") in (None, "")


def test_long_always_wrong_early_stop(tmp_path):
    payload = _dev_run(tmp_path, {"run": "long", "mode": "fail", "seed": 12})
    assert LONG_EARLY_STOP_AFTER <= payload["steps"] < CAP_LONG
    assert payload.get("open_used", 0) == 0
    assert payload.get("stop_reason") == "fail_fast_long"
    assert payload.get("ff_win_obj_len_short") == SHORT_FAIL_FAST_WINDOW
    assert payload.get("ff_win_obj_len_long") == LONG_FAIL_FAST_WINDOW


def test_long_fail_blocks_open(tmp_path):
    payload = _dev_run(tmp_path, {"run": "long", "mode": "fail", "seed": 12})
    assert payload["steps"] < CAP_LONG
    assert payload.get("stop_reason") == "fail_fast_long"
    assert payload.get("open_used", 0) == 0
    assert payload.get("sr_used", 0) == len(DOMAINS) * SR_PER_DOMAIN_LONG
    assert payload.get("ff_win_obj_len_short") == SHORT_FAIL_FAST_WINDOW
    assert payload.get("ff_win_obj_len_long") == LONG_FAIL_FAST_WINDOW
    reasons = payload.get("reasons") or {}
    reason_values = []
    for entry in reasons.values():
        if isinstance(entry, list):
            reason_values.extend(entry)
        elif entry:
            reason_values.append(entry)
    assert "open_blocked_progress" in reason_values
    probe_meta = payload.get("god_probe") or {}
    for domain in DOMAINS:
        domain_meta = probe_meta.get(domain) or {}
        if domain_meta.get("probe_reason") is not None:
            assert domain_meta.get("probe_reason") == "blocked_fail_fast"


def test_long_pass_unchanged(tmp_path):
    payload = _dev_run(tmp_path, {"run": "long", "mode": "pass", "seed": 11})
    assert payload["steps"] == CAP_LONG
    assert payload.get("open_used", 0) >= 8
    assert payload.get("stop_reason") in (None, "")


def test_fail_fast_windows_ignore_sr(tmp_path):
    session = AdaptiveSession("long")

    for _ in range(5):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    session._record_fail_fast_stats(
        is_objective=False,
        correct=True,
        info_gain=0.2,
    )

    for _ in range(5):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    assert session.ff_win_obj_len_short == 10
    assert session.ff_win_obj_len_long == 10
    assert session.rolling_correct(LONG_FAIL_FAST_WINDOW) == 0


def test_fail_fast_windows_no_double_count(tmp_path):
    session = AdaptiveSession("short")

    session._record_fail_fast_stats(
        is_objective=True,
        correct=False,
        info_gain=0.0,
    )
    session._record_fail_fast_stats(
        is_objective=True,
        correct=False,
        info_gain=0.0,
    )

    assert session.ff_win_obj_len_short == 2
    assert session.ff_win_obj_len_long in (0, 2)
    assert session.rolling_correct(SHORT_FAIL_FAST_WINDOW) == 0
    assert session.median_info(LONG_FAIL_FAST_WINDOW) >= 0.0


def test_short_long_fail_fast_window_split(tmp_path):
    session = AdaptiveSession("long")

    for _ in range(8):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=True,
            info_gain=0.05,
        )

    for _ in range(12):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    assert session.ff_win_obj_len_short == SHORT_FAIL_FAST_WINDOW
    assert session.ff_win_obj_len_long == LONG_FAIL_FAST_WINDOW

    short_sum = session.rolling_correct(SHORT_FAIL_FAST_WINDOW)
    long_sum = session.rolling_correct(LONG_FAIL_FAST_WINDOW)
    assert short_sum == 0
    assert long_sum == 8

    policy_short = QuestionPolicy([], "short")
    hist_short = {
        d: DomainHistory(obj_count=SHORT_OBJ_MIN, sr_count=SHORT_SR_PER_DOMAIN)
        for d in DOMAINS
    }
    state_short = PolicyState(
        run_type="short",
        theta={d: 0.0 for d in DOMAINS},
        se={d: 0.3 for d in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=(SHORT_OBJ_MIN + SHORT_SR_PER_DOMAIN) * len(DOMAINS),
        hist=hist_short,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type=None,
        rolling_correct_short=short_sum,
        rolling_correct_long=0,
        median_info_long=0.0,
        rolling_info_long=0.0,
        fail_fast_long_active=False,
        ff_len_short=session.ff_win_obj_len_short,
        ff_len_long=session.ff_win_obj_len_long,
        sr_done=True,
    )
    assert policy_short.should_stop(state_short) is True
    assert policy_short.stop_reason == "fail_fast_short"

    policy_long = QuestionPolicy([], "long")
    hist_long = {
        d: DomainHistory(
            obj_count=OBJ_MIN_LONG,
            sr_count=SR_PER_DOMAIN_LONG,
            obj_done=True,
        )
        for d in DOMAINS
    }
    state_long = PolicyState(
        run_type="long",
        theta={d: 0.0 for d in DOMAINS},
        se={d: 0.6 for d in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=LONG_EARLY_STOP_AFTER,
        hist=hist_long,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type=None,
        rolling_correct_short=0,
        rolling_correct_long=long_sum,
        median_info_long=0.0,
        rolling_info_long=0.0,
        fail_fast_long_active=False,
        ff_len_short=session.ff_win_obj_len_short,
        ff_len_long=session.ff_win_obj_len_long,
        sr_done=True,
    )
    assert policy_long._long_fail_fast_signal(state_long, sr_needed=0) is False

    for _ in range(8):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    updated_long_sum = session.rolling_correct(LONG_FAIL_FAST_WINDOW)
    assert updated_long_sum == 0
    state_long_after = PolicyState(
        run_type="long",
        theta={d: 0.0 for d in DOMAINS},
        se={d: 0.6 for d in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=LONG_EARLY_STOP_AFTER + 1,
        hist=hist_long,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type=None,
        rolling_correct_short=0,
        rolling_correct_long=updated_long_sum,
        median_info_long=0.0,
        rolling_info_long=0.0,
        fail_fast_long_active=False,
        ff_len_short=session.ff_win_obj_len_short,
        ff_len_long=session.ff_win_obj_len_long,
        sr_done=True,
    )
    assert policy_long._long_fail_fast_signal(state_long_after, sr_needed=0) is True


def test_long_fail_fast_mixed_sequence(tmp_path):
    session = AdaptiveSession("long")

    for _ in range(8):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=True,
            info_gain=0.04,
        )

    for _ in range(5):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    oscillating_tail = [
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    for flag in oscillating_tail:
        session._record_fail_fast_stats(
            is_objective=True,
            correct=flag,
            info_gain=0.04 if flag else 0.0,
        )

    assert session.ff_win_obj_len_short == SHORT_FAIL_FAST_WINDOW
    assert session.ff_win_obj_len_long == LONG_FAIL_FAST_WINDOW

    short_sum = session.rolling_correct(SHORT_FAIL_FAST_WINDOW)
    long_sum = session.rolling_correct(LONG_FAIL_FAST_WINDOW)
    assert short_sum <= SHORT_FAIL_FAST_MAX_CORRECT
    assert long_sum > LONG_FAIL_FAST_MAX_CORRECT

    median_info = session.median_info(LONG_FAIL_FAST_WINDOW)
    assert median_info < LONG_MIN_INFO_GAIN

    policy = QuestionPolicy([], "long")
    hist = {
        d: DomainHistory(
            obj_count=OBJ_MIN_LONG,
            sr_count=SR_PER_DOMAIN_LONG,
            open_count=0,
            obj_done=True,
        )
        for d in DOMAINS
    }

    state_before = PolicyState(
        run_type="long",
        theta={d: 0.0 for d in DOMAINS},
        se={d: 0.6 for d in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=LONG_EARLY_STOP_AFTER + 1,
        hist=hist,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type="MCQ",
        rolling_correct_short=short_sum,
        rolling_correct_long=long_sum,
        median_info_long=median_info,
        rolling_info_long=median_info,
        fail_fast_long_active=False,
        ff_len_short=session.ff_win_obj_len_short,
        ff_len_long=session.ff_win_obj_len_long,
        sr_done=True,
    )

    assert policy._long_fail_fast_signal(state_before, sr_needed=0) is False
    assert policy.should_stop(state_before) is False
    assert policy.stop_reason is None

    for _ in range(6):
        session._record_fail_fast_stats(
            is_objective=True,
            correct=False,
            info_gain=0.0,
        )

    assert session.ff_win_obj_len_short == SHORT_FAIL_FAST_WINDOW
    assert session.ff_win_obj_len_long == LONG_FAIL_FAST_WINDOW

    short_sum_after = session.rolling_correct(SHORT_FAIL_FAST_WINDOW)
    long_sum_after = session.rolling_correct(LONG_FAIL_FAST_WINDOW)
    median_after = session.median_info(LONG_FAIL_FAST_WINDOW)

    assert short_sum_after <= SHORT_FAIL_FAST_MAX_CORRECT
    assert long_sum_after <= LONG_FAIL_FAST_MAX_CORRECT
    assert median_after < LONG_MIN_INFO_GAIN

    state_after = PolicyState(
        run_type="long",
        theta={d: 0.0 for d in DOMAINS},
        se={d: 0.6 for d in DOMAINS},
        asked=set(),
        seen_variant_groups=set(),
        step=LONG_EARLY_STOP_AFTER + 10,
        hist=hist,
        info_history=[],
        mirrored_domains_planned=set(),
        last_domain_id=None,
        last_item_type="MCQ",
        rolling_correct_short=short_sum_after,
        rolling_correct_long=long_sum_after,
        median_info_long=median_after,
        rolling_info_long=median_after,
        fail_fast_long_active=False,
        ff_len_short=session.ff_win_obj_len_short,
        ff_len_long=session.ff_win_obj_len_long,
        sr_done=True,
    )

    assert policy.should_stop(state_after) is True
    assert policy.stop_reason == "fail_fast_long"
