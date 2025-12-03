import importlib
import os
import sys
from typing import Callable

from fastapi.testclient import TestClient

from skill_core.config import GLOBAL_STEP_CAP, CAP_SHORT, CAP_LONG

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


def _answer_value(item, session=None):
    itype = item.get("type")
    if session is not None:
        obj = getattr(session, "_id_to_item", {}).get(item.get("id"))
        if obj is not None:
            t = str(getattr(obj, "type", itype or "")).upper()
            if t == "MCQ":
                correct = getattr(obj, "correct", None)
                if isinstance(correct, int):
                    return correct
            if t == "SJT":
                keys = getattr(obj, "keys", None) or getattr(obj, "sjt_keys", None)
                if isinstance(keys, dict) and keys:
                    try:
                        return int(max(
                            ((int(k), float(v)) for k, v in keys.items()),
                            key=lambda kv: kv[1],
                        )[0])
                    except Exception:
                        pass
                for field in ("best_index", "best", "key_best"):
                    val = getattr(obj, field, None)
                    if isinstance(val, int):
                        return int(val)
            if t == "SR":
                return 4
            if t == "OPEN":
                return (
                    "Deliver a 3-step plan with 90% accuracy metric, weekly review cadence, validation baseline, "
                    "and a go/no-go decision gate."
                )
    if itype == "OPEN":
        return (
            "Deliver a 3-step plan with 90% accuracy metric, weekly review cadence, validation baseline, "
            "and a go/no-go decision gate."
        )
    if itype == "MCQ":
        correct = item.get("correct")
        if isinstance(correct, int):
            return correct
        try:
            return int(correct)
        except Exception:
            return 1
    if itype == "SJT":
        keys = item.get("keys") or item.get("sjt_keys")
        if isinstance(keys, dict) and keys:
            try:
                return int(max(keys.items(), key=lambda kv: float(kv[1]))[0])
            except Exception:
                pass
        for field in ("best_index", "best", "key_best"):
            if isinstance(item.get(field), int):
                return int(item[field])
        return 1
    return 3


def _wrong_answer_value(item, session=None):
    itype = item.get("type")
    if itype in {"MCQ", "SJT"}:
        return -1
    if itype == "OPEN":
        return "Declining to provide details."
    return 1


def _run_session_flow(
    client: TestClient,
    app_module,
    run_type: str,
    answer_fn: Callable[[dict, object], int | str],
):
    start = client.post("/session/start", json={"run": run_type, "llm": "none"})
    assert start.status_code == 200
    sid = start.json()["session_id"]
    session = app_module.SESS.get(sid)

    steps = 0
    while True:
        nxt = client.get(f"/session/{sid}/next")
        if nxt.status_code == 204:
            break
        assert nxt.status_code == 200
        item = nxt.json().get("item")
        assert item, "Expected item payload"

        answer_val = answer_fn(item, session)
        payload = {
            "item_id": item["id"],
            "value": answer_val,
            "rt_ms": 2500 if item.get("type") == "OPEN" else 2000,
        }
        ans = client.post(f"/session/{sid}/answer", json=payload)
        assert ans.status_code == 200
        steps += 1
        if ans.json().get("done"):
            break
        assert steps <= GLOBAL_STEP_CAP

    finish = client.post("/session/finish", json={"session_id": sid})
    assert finish.status_code == 200
    return finish.json(), steps


def test_short_end_to_end_ok(tmp_path):
    _, app_module = _reload_app(tmp_path)
    client = TestClient(app_module.app)

    start = client.post("/session/start", json={"run": "short", "llm": "none"})
    assert start.status_code == 200
    payload = start.json()
    sid = payload["session_id"]

    steps = 0
    while True:
        nxt = client.get(f"/session/{sid}/next")
        if nxt.status_code == 204:
            break
        assert nxt.status_code == 200
        item = nxt.json().get("item")
        assert item
        payload = {"item_id": item["id"], "value": _answer_value(item)}
        if item.get("type") == "OPEN":
            payload["rt_ms"] = 2500
        else:
            payload["rt_ms"] = 2000
        ans = client.post(f"/session/{sid}/answer", json=payload)
        assert ans.status_code == 200
        steps += 1
        if ans.json().get("done"):
            break
        assert steps <= GLOBAL_STEP_CAP

    finish = client.post("/session/finish", json={"session_id": sid})
    assert finish.status_code == 200
    report = finish.json()
    tiers = [str(d.get("tier", "")) for d in report.get("domain_scores", [])]
    assert tiers, "Expected domain tiers in report"
    assert all(tier in {"F", "D", "C", "B", "A"} for tier in tiers)

    summary = report.get("summary") or {}
    assert summary.get("steps_used") == steps
    cap_str = summary.get("cap", "")
    assert cap_str == f"{CAP_SHORT}/{CAP_LONG}"
    assert summary.get("sr_used", 0) >= 0
    assert summary.get("open_used", 0) == 0
def test_long_finish_requires_done(tmp_path):
    _, app_module = _reload_app(tmp_path)
    client = TestClient(app_module.app)

    start = client.post("/session/start", json={"run": "long", "llm": "none"})
    assert start.status_code == 200
    sid = start.json()["session_id"]

    nxt = client.get(f"/session/{sid}/next")
    assert nxt.status_code == 200
    item = nxt.json().get("item")
    assert item
    payload = {"item_id": item["id"], "value": _answer_value(item), "rt_ms": 2000}
    ans = client.post(f"/session/{sid}/answer", json=payload)
    assert ans.status_code == 200

    attempt = client.post("/session/finish", json={"session_id": sid})
    assert attempt.status_code == 409
    assert attempt.json()["status"] == "continue"

    steps = 1
    while True:
        nxt = client.get(f"/session/{sid}/next")
        if nxt.status_code == 204:
            break
        assert nxt.status_code == 200
        item = nxt.json().get("item")
        assert item
        payload = {"item_id": item["id"], "value": _answer_value(item)}
        if item.get("type") == "OPEN":
            payload["rt_ms"] = 2500
        else:
            payload["rt_ms"] = 2000
        ans = client.post(f"/session/{sid}/answer", json=payload)
        assert ans.status_code == 200
        steps += 1
        if ans.json().get("done"):
            break
        assert steps <= GLOBAL_STEP_CAP

    finish = client.post("/session/finish", json={"session_id": sid})
    assert finish.status_code == 200
    report = finish.json()
    summary = report.get("summary") or {}
    assert summary.get("steps_used") == steps
    cap_str = summary.get("cap", "")
    assert cap_str == f"{CAP_LONG}/{CAP_LONG}", cap_str
    used_val = summary.get("steps_used")
    limit_val = CAP_LONG
    assert summary.get("sr_used") == 16
    open_used = summary.get("open_used", 0)
    assert open_used == 0 or open_used >= 8
    assert used_val <= GLOBAL_STEP_CAP


def test_finish_summary_stop_reasons_and_fail_fast_windows(tmp_path):
    os.environ["STAGING_PROFILE"] = "1"
    try:
        pass_overrides = {
            "LONG_EARLY_STOP_ENABLE": "0",
            "SHORT_FAIL_FAST_MAX_CORRECT": "-1",
            "SHORT_EXTRAS_REQUIRE_PROGRESS": "0",
            "SHORT_EXTRAS_SE_MAX": "1.0",
            "TEST_MODE": "1",
        }
        for key, value in pass_overrides.items():
            os.environ[key] = value

        _, app_module = _reload_app(tmp_path)
        client_pass = TestClient(app_module.app)

        short_pass_report, short_pass_steps = _run_session_flow(client_pass, app_module, "short", _answer_value)
        long_pass_report, long_pass_steps = _run_session_flow(client_pass, app_module, "long", _answer_value)

        for key in pass_overrides:
            os.environ.pop(key, None)

        os.environ["LONG_EARLY_STOP_ENABLE"] = "1"
        _, app_module = _reload_app(tmp_path)
        client_fail = TestClient(app_module.app)

        short_fail_report, short_fail_steps = _run_session_flow(client_fail, app_module, "short", _wrong_answer_value)
        long_fail_report, long_fail_steps = _run_session_flow(client_fail, app_module, "long", _wrong_answer_value)

        def _summary(report):
            return report.get("summary") or {}

        short_pass_summary = _summary(short_pass_report)
        assert short_pass_summary.get("stop_reason") in (None, "")
        assert short_pass_summary.get("steps_used") == short_pass_steps == 40
        assert short_pass_summary.get("sr_used") == 8
        assert short_pass_summary.get("open_used") == 0
        assert short_pass_summary.get("cap") == f"{CAP_SHORT}/{CAP_LONG}"
        assert short_pass_summary.get("effective_steps") == short_pass_steps
        assert short_pass_summary.get("ff_win_obj_len_short") == 12
        assert short_pass_summary.get("ff_win_obj_len_long") in (0, 20)

        short_fail_summary = _summary(short_fail_report)
        assert short_fail_summary.get("stop_reason") == "fail_fast_short"
        assert short_fail_summary.get("steps_used") == short_fail_steps
        assert short_fail_summary.get("steps_used") < 40
        assert short_fail_summary.get("sr_used") == 8
        assert short_fail_summary.get("open_used") == 0
        assert short_fail_summary.get("cap") == f"{CAP_SHORT}/{CAP_LONG}"
        assert short_fail_summary.get("effective_steps") == short_fail_steps
        assert short_fail_summary.get("ff_win_obj_len_short") == 12
        assert short_fail_summary.get("ff_win_obj_len_long") in (0, 20)

        long_pass_summary = _summary(long_pass_report)
        assert long_pass_summary.get("stop_reason") in (None, "")
        assert long_pass_summary.get("steps_used") == long_pass_steps
        assert 100 <= long_pass_steps <= CAP_LONG
        assert long_pass_summary.get("sr_used") == 16
        assert long_pass_summary.get("open_used") >= 8
        assert long_pass_summary.get("cap") == f"{CAP_LONG}/{CAP_LONG}"
        assert long_pass_summary.get("effective_steps") == long_pass_summary.get(
            "steps_used"
        )
        assert long_pass_summary.get("ff_win_obj_len_short") == 12

        long_fail_summary = _summary(long_fail_report)
        assert long_fail_summary.get("stop_reason") in (None, "")
        assert long_fail_summary.get("steps_used") == long_fail_steps
        assert long_fail_summary.get("steps_used") < CAP_LONG
        assert long_fail_summary.get("sr_used") == 16
        assert long_fail_summary.get("open_used") == 0
        assert long_fail_summary.get("cap") == f"{CAP_LONG}/{CAP_LONG}"
        assert long_fail_summary.get("effective_steps") == long_fail_steps
        assert long_fail_summary.get("effective_steps") == long_fail_summary.get(
            "steps_used"
        )
        assert long_fail_summary.get("ff_win_obj_len_short") == 12
    finally:
        for key in ("STAGING_PROFILE", "LONG_EARLY_STOP_ENABLE"):
            os.environ.pop(key, None)
        _reload_app(tmp_path)


def test_dev_run_modes(tmp_path):
    os.environ["STAGING_PROFILE"] = "1"
    try:
        _, app_module = _reload_app(tmp_path)
        client = TestClient(app_module.app)

        short_resp = client.get("/dev/run", params={"run": "short", "mode": "pass", "seed": 11})
        assert short_resp.status_code == 200
        short_payload = short_resp.json()
        assert short_payload["steps"] == 40
        assert len(short_payload["tiers"]) == 8
        assert all(tier in {"F", "D", "C", "B", "A"} for tier in short_payload["tiers"])

        long_pass = client.get("/dev/run", params={"run": "long", "mode": "pass", "seed": 11})
        assert long_pass.status_code == 200
        long_payload = long_pass.json()
        assert str(long_payload.get("cap", "")).endswith("/160")
        assert long_payload.get("sr_used") == 16
        assert int(long_payload.get("open_used", 0)) >= 8
        assert 100 <= int(long_payload.get("steps", 0)) <= CAP_LONG
        assert long_payload.get("effective_steps") == long_payload.get("steps")

        long_fail = client.get("/dev/run", params={"run": "long", "mode": "fail", "seed": 11})
        assert long_fail.status_code == 200
        fail_payload = long_fail.json()
        assert int(fail_payload.get("open_used", 0)) == 0
        assert int(fail_payload.get("steps", 0)) <= CAP_LONG
        assert fail_payload.get("effective_steps") == fail_payload.get("steps")
    finally:
        os.environ.pop("STAGING_PROFILE", None)
        _reload_app(tmp_path)


def test_dev_run_force_open(tmp_path):
    os.environ["STAGING_PROFILE"] = "1"
    os.environ["TEST_MODE"] = "1"
    os.environ["TEST_OPEN_FULL"] = "1"
    try:
        _, app_module = _reload_app(tmp_path)
        client = TestClient(app_module.app)

        forced = client.get("/dev/run", params={"run": "long", "mode": "pass", "seed": 11})
        assert forced.status_code == 200
        payload = forced.json()
        assert payload.get("sr_used") == 16
        assert payload.get("open_used") == 16
        assert int(payload.get("steps", 0)) <= CAP_LONG
        assert payload.get("effective_steps") == payload.get("steps")
    finally:
        for key in ("STAGING_PROFILE", "TEST_MODE", "TEST_OPEN_FULL"):
            os.environ.pop(key, None)
        _reload_app(tmp_path)
