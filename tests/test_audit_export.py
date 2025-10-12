from __future__ import annotations

import importlib
import json
import os
import sys

from fastapi.testclient import TestClient

from tests.conftest import build_synthetic_bank


_DEF_MODULES = [
    "skill_core.config",
    "api.storage",
    "api.app",
]


def _reload_app(tmp_path) -> tuple[object, object]:
    os.environ["DATA_DIR"] = str(tmp_path)
    for name in _DEF_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            __import__(name)
    storage = sys.modules["api.storage"]
    app_module = sys.modules["api.app"]
    return storage, app_module


def test_audit_exports_available(tmp_path, monkeypatch):
    _storage, app_module = _reload_app(tmp_path / "enabled")
    import skill_core.question_bank as qb

    monkeypatch.setattr(qb, "load_bank", lambda: build_synthetic_bank())

    client = TestClient(app_module.app)

    start = client.post("/session/start", json={"run": "long", "llm": "none"})
    assert start.status_code == 200
    sid = start.json()["session_id"]

    first = client.get("/api/test/next", params={"session_id": sid})
    assert first.status_code == 200
    item = first.json()["item"]
    assert item

    answer_payload = {
        "session_id": sid,
        "item_id": item["id"],
        "answer": 0 if item["type"] in {"MCQ", "SJT"} else "4",
        "started_at": 0.0,
        "submitted_at": 1.0,
    }
    resp_answer = client.post("/api/test/answer", json=answer_payload)
    assert resp_answer.status_code == 200

    finish = client.post("/api/test/finish", json={"session_id": sid})
    assert finish.status_code == 200
    report = finish.json()
    report_id = report.get("reportId") or report.get("id")
    assert report_id

    json_resp = client.get(f"/results/{report_id}/audit.json")
    assert json_resp.status_code == 200
    body = json_resp.json()
    events = body.get("events") or []
    assert events, "expected at least one audit event"
    first_event = events[0]
    required = {
        "t",
        "domain",
        "item_id",
        "type",
        "b",
        "level_before",
        "level_after",
        "correct_or_r",
        "theta_before",
        "theta_after",
        "se_after",
        "latency_ms",
    }
    assert required.issubset(first_event.keys())

    csv_resp = client.get(f"/results/{report_id}/audit.csv")
    assert csv_resp.status_code == 200
    csv_lines = [line for line in csv_resp.text.strip().splitlines() if line]
    assert len(csv_lines) == len(events) + 1
    header = csv_lines[0].split(",")
    assert header[0] == "t"
    assert header[-1] == "latency_ms"


def test_audit_exports_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_EXPORT_ENABLED", "0")
    storage, app_module = _reload_app(tmp_path / "disabled")

    payload = {
        "run_type": "long",
        "audit_events": [
            {
                "t": "2024-01-01T00:00:00+00:00",
                "domain": "Analytical",
                "item_id": "x1",
                "type": "MCQ",
                "b": 0,
                "level_before": 0,
                "level_after": 0,
                "correct_or_r": 1.0,
                "theta_before": 0.0,
                "theta_after": 0.1,
                "se_after": 0.9,
                "latency_ms": 1200,
            }
        ],
        "meta": {},
        "summary": {},
    }
    report_id = "audit-disabled"
    path = storage.REPORTS_DIR / f"{report_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(app_module.app)

    json_resp = client.get(f"/results/{report_id}/audit.json")
    csv_resp = client.get(f"/results/{report_id}/audit.csv")
    assert json_resp.status_code == 404
    assert csv_resp.status_code == 404
