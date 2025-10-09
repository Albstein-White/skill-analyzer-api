from __future__ import annotations

import importlib
import json
import os
import sys

from fastapi.testclient import TestClient

from skill_core.config import PLAN_BULLETS_MAX, PLAN_BULLETS_MIN, PLAN_FOCUS_MAX


def _reload_app(tmp_path) -> tuple[object, object]:
    os.environ["DATA_DIR"] = str(tmp_path)
    if "api.storage" in sys.modules:
        importlib.reload(sys.modules["api.storage"])
    else:
        import api.storage  # noqa: F401
    storage = sys.modules["api.storage"]
    if "api.app" in sys.modules:
        importlib.reload(sys.modules["api.app"])
    else:
        import api.app  # noqa: F401
    app_module = sys.modules["api.app"]
    return storage, app_module


def _write_report(storage, report_id: str, payload: dict) -> None:
    path = storage.REPORTS_DIR / f"{report_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_plan_endpoint_idempotent(tmp_path):
    storage, app_module = _reload_app(tmp_path)
    client = TestClient(app_module.app)

    long_payload = {
        "run_type": "long",
        "summary": {"total_items": 120},
        "meta": {"run": "long", "total_items": 120},
        "domain_scores": [
            {
                "domain": "Analytical",
                "tier": "D",
                "theta": -0.2,
                "se": 0.35,
                "b_stable": 0,
                "items_by_level": {0: 8},
                "accuracy_by_level": {0: 0.5},
            },
            {
                "domain": "Mathematical",
                "tier": "C",
                "theta": 0.1,
                "se": 0.28,
                "b_stable": 1,
                "items_by_level": {1: 10},
                "accuracy_by_level": {1: 0.6},
            },
            {
                "domain": "Verbal",
                "tier": "B",
                "theta": 0.4,
                "se": 0.22,
                "b_stable": 1,
                "items_by_level": {1: 9},
                "accuracy_by_level": {1: 0.7},
            },
        ],
    }

    _write_report(storage, "report-long", long_payload)

    resp1 = client.post("/results/report-long/plan")
    assert resp1.status_code == 200
    body1 = resp1.json()
    plan = body1.get("plan")
    assert plan and 2 <= len(plan) <= PLAN_FOCUS_MAX
    for entry in plan:
        goals = entry.get("goals", [])
        assert PLAN_BULLETS_MIN <= len(goals) <= PLAN_BULLETS_MAX

    resp2 = client.post("/results/report-long/plan")
    assert resp2.status_code == 200
    assert resp2.json()["plan"] == plan, "Subsequent calls should return cached plan"

    short_payload = {
        "run_type": "short",
        "summary": {"total_items": 40},
        "meta": {"run": "short", "total_items": 40},
        "domain_scores": long_payload["domain_scores"],
    }
    _write_report(storage, "report-short", short_payload)

    resp_short = client.post("/results/report-short/plan")
    assert resp_short.status_code == 200
    assert resp_short.json().get("plan") == []

