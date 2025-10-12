import importlib
import os
import sys

from fastapi.testclient import TestClient

from skill_core.config import GLOBAL_STEP_CAP

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


def _answer_value(item):
    itype = item.get("type")
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
    assert cap_str.startswith(f"{steps}/")
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
    assert cap_str, "cap summary should be present"
    used_str, _, limit_str = cap_str.partition("/")
    used_val = int(used_str)
    limit_val = int(limit_str)
    assert used_val == summary.get("steps_used")
    assert limit_val == GLOBAL_STEP_CAP
    assert summary.get("sr_used") == 16
    assert summary.get("open_used") >= 8
    assert used_val <= GLOBAL_STEP_CAP


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

        long_fail = client.get("/dev/run", params={"run": "long", "mode": "fail", "seed": 11})
        assert long_fail.status_code == 200
        fail_payload = long_fail.json()
        assert int(fail_payload.get("open_used", 0)) <= 8
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
    finally:
        for key in ("STAGING_PROFILE", "TEST_MODE", "TEST_OPEN_FULL"):
            os.environ.pop(key, None)
        _reload_app(tmp_path)
