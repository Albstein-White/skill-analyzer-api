import importlib
import os
import sys

from fastapi.testclient import TestClient

from skill_core.config import (
    CAP_SHORT,
    SHORT_FAIL_FAST_MAX_CORRECT,
    SHORT_FAIL_FAST_WINDOW,
    SHORT_OBJ_MIN,
    SHORT_SR_PER_DOMAIN,
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
    assert session.ff_win_obj_len_long == 0
    assert session.rolling_correct(SHORT_FAIL_FAST_WINDOW) == 0


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
    assert session.rolling_correct(SHORT_FAIL_FAST_WINDOW) == 0


def test_short_fail_fast_window_split(tmp_path):
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
    assert session.ff_win_obj_len_long == 0

    short_sum = session.rolling_correct(SHORT_FAIL_FAST_WINDOW)
    assert short_sum == 0

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
    )
    assert policy_short._should_stop_short(state_short)
